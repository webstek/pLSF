// ****************************************************************************
/// @file main.cpp
/// @author Kyle Webster
/// @version 0.1
/// @date 12 Apr 2026
/// @brief pLSF – Parallelized Lower-Star Filtration
/// @details CLI entry point: parse arguments → load volume → run filtration
// ****************************************************************************

#include "boundary.hpp"
#include "grid.hpp"
#include "io.hpp"
#include "lsf.hpp"
#include "sycl_utils.hpp"

#include <phat/compute_persistence_pairs.h>
#include <phat/persistence_pairs.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Config
{
  std::string input_path;
  std::string output_stem; ///< stem for .bin/.vals output; empty = no output
  std::string device_hint = "gpu"; ///< "cpu" | "gpu" | "default"
  bool        verbose = false;
  bool        timings = false;
  bool        pairs = false; ///< compute persistence pairs via phat
  bool filtration_only
      = false; ///< stop after filtration; do not compute boundary matrix
};

static void print_usage (const char *prog)
{
  std::cout << "Usage: " << prog
            << " [options] <input.nii>\n"
               "\n"
               "Options:\n"
               "  -d, --device  <mode>  SYCL device selector: cpu|gpu|default  "
               "[gpu]\n"
               "  -o, --output  <stem>  Write output files with the given "
               "stem\n"
               "  -p, --pairs           Compute persistence pairs via phat\n"
               "                        (default: output boundary matrix "
               "only)\n"
               "  -f, --filtration-only Stop after filtration is computed\n"
               "                        (skip boundary matrix and output)\n"
               "  -t, --timings         Report per-step wall time and memory "
               "usage  [off]\n"
               "  -v, --verbose         Print device info and grid statistics  "
               " [off]\n"
               "  -h, --help            Show this message and exit\n"
               "\n"
               "Output (requires -o):\n"
               "  Without -p:  <stem>.bin   PHAT binary boundary matrix\n"
               "               <stem>.vals  Filtration values (binary "
               "doubles)\n"
               "  With -p:     <stem>.pairs Persistence pairs (PHAT binary)\n"
               "               <stem>.vals  Filtration values (binary "
               "doubles)\n"
               "\n"
               "Supported formats: .nii (NIfTI-1/2, uncompressed)\n";
}

static Config parse_args (int argc, char *argv[])
{
  if (argc < 2)
    {
      print_usage (argv[0]);
      std::exit (1);
    }

  Config cfg;
  for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      if (a == "-h" || a == "--help")
        {
          print_usage (argv[0]);
          std::exit (0);
        }
      else if (a == "-v" || a == "--verbose")
        cfg.verbose = true;
      else if (a == "-t" || a == "--timings")
        cfg.timings = true;
      else if (a == "-p" || a == "--pairs")
        cfg.pairs = true;
      else if (a == "-f" || a == "--filtration-only")
        cfg.filtration_only = true;
      else if ((a == "-d" || a == "--device") && i + 1 < argc)
        cfg.device_hint = argv[++i];
      else if ((a == "-o" || a == "--output") && i + 1 < argc)
        cfg.output_stem = argv[++i];
      else if (a[0] != '-')
        cfg.input_path = a;
      else
        {
          std::cerr << "Unknown option: " << a << '\n';
          std::exit (1);
        }
    }

  if (cfg.input_path.empty ())
    {
      std::cerr << "Error: no input file specified.\n";
      std::exit (1);
    }
  if (cfg.filtration_only && cfg.pairs)
    {
      std::cerr
          << "Error: --pairs cannot be used with --filtration-only.\n";
      std::exit (1);
    }
  if (cfg.filtration_only && !cfg.output_stem.empty ())
    {
      std::cerr
          << "Error: --output cannot be used with --filtration-only.\n";
      std::exit (1);
    }
  return cfg;
}

// ****************************************************************************
// Timing and memory helpers

using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

struct TimingRow
{
  const char *label;
  double      ms;
  long        host_rss_kb; ///< resident set size after step; -1 = unavailable
  long        dev_peak_kb; ///< analytical peak device allocation; -1 = N/A
};

static long cur_host_rss_kb ()
{
#ifdef __linux__
  std::FILE *f = std::fopen ("/proc/self/status", "r");
  if (!f)
    return -1;
  char line[128];
  long rss = -1;
  while (std::fgets (line, sizeof line, f))
    if (std::sscanf (line, "VmRSS: %ld kB", &rss) == 1)
      break;
  std::fclose (f);
  return rss;
#else
  return -1;
#endif
}

static std::string fmt_mem (long kb)
{
  if (kb < 0)
    return "-";
  if (kb < 1024)
    return std::to_string (kb) + " KiB";
  if (kb < 1024 * 1024)
    return std::to_string (kb / 1024) + " MiB";
  return std::to_string (kb / (1024 * 1024)) + " GiB";
}

static void print_timings (const std::vector<TimingRow> &rows)
{
  constexpr int W_LABEL = 26;
  constexpr int W_TIME = 12;
  constexpr int W_MEM = 13;
  const int     W_TOTAL = W_LABEL + W_TIME + W_MEM + W_MEM;

  auto row_line = [&] (const char *label, const std::string &time_s,
                      long host_kb, long dev_kb) {
    std::cout << std::left << std::setw (W_LABEL) << label << std::right
              << std::setw (W_TIME) << time_s << std::setw (W_MEM)
              << fmt_mem (host_kb) << std::setw (W_MEM) << fmt_mem (dev_kb)
              << '\n';
  };

  auto fmt_ms = [] (double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision (1) << ms << " ms";
    return ss.str ();
  };

  std::cout << '\n'
            << std::left << std::setw (W_LABEL) << "Step" << std::right
            << std::setw (W_TIME) << "Time" << std::setw (W_MEM) << "Host RSS"
            << std::setw (W_MEM) << "Dev peak" << '\n'
            << std::string (W_TOTAL, '-') << '\n';

  double total = 0;
  for (auto const &r : rows)
    {
      total += r.ms;
      row_line (r.label, fmt_ms (r.ms), r.host_rss_kb, r.dev_peak_kb);
    }

  std::cout << std::string (W_TOTAL, '-') << '\n';
  std::cout << std::left << std::setw (W_LABEL) << "Total" << std::right
            << std::setw (W_TIME) << fmt_ms (total) << '\n'
            << '\n';
}

// ****************************************************************************
int main (int argc, char *argv[])
{
  const Config cfg = parse_args (argc, argv);

  // Select SYCL device
  plsf::DevicePreference dev_pref = plsf::DevicePreference::GPU;
  if (cfg.device_hint == "cpu")
    dev_pref = plsf::DevicePreference::CPU;
  else if (cfg.device_hint == "default")
    dev_pref = plsf::DevicePreference::Default;

  sycl::queue queue = plsf::make_queue (dev_pref);
  if (cfg.verbose)
    plsf::print_device_info (queue);

  std::vector<TimingRow> rows;

  try
    {
      // NIfTI load
      if (cfg.verbose)
        std::cout << "Loading : " << cfg.input_path << '\n';

      auto              t0 = Clock::now ();
      plsf::Grid<float> grid = plsf::read_nifti<float> (cfg.input_path);
      rows.push_back ({ "NIfTI load", Ms (Clock::now () - t0).count (),
          cur_host_rss_kb (), -1 });

      if (cfg.verbose)
        std::cout << "Grid    : " << grid.Nx << " x " << grid.Ny << " x "
                  << grid.Nz << "  (" << grid.Nx * grid.Ny * grid.Nz
                  << " cells total)\n";

      // Precompute grid dimensions for device memory accounting
      const uint64_t Nx = static_cast<uint64_t> (grid.Nx);
      const uint64_t Ny = static_cast<uint64_t> (grid.Ny);
      const uint64_t Nz = static_cast<uint64_t> (grid.Nz);
      const uint64_t n_grid = Nx * Ny * Nz;
      const uint64_t n_cells = (2 * Nx - 1) * (2 * Ny - 1) * (2 * Nz - 1);

      plsf::LowerStarFiltration<float> filtration (grid);

      // Complex computation
      t0 = Clock::now ();
      filtration.compute_complex (queue);
      {
        // Peak device: d_grid and d_cube_map live simultaneously
        const long dev_kb
            = static_cast<long> ((n_grid + n_cells) * sizeof (float) / 1024);
        rows.push_back ({ "Complex computation",
            Ms (Clock::now () - t0).count (), cur_host_rss_kb (), dev_kb });
      }

      // Complex sorting
      t0 = Clock::now ();
      filtration.compute_ordering (queue);
      {
        // Peak device: d_dims only
        const long dev_kb = static_cast<long> (n_cells * sizeof (int) / 1024);
        rows.push_back ({ "Complex sorting", Ms (Clock::now () - t0).count (),
            cur_host_rss_kb (), dev_kb });
      }

      if (cfg.filtration_only)
        {
          if (cfg.verbose)
            std::cout
                << "Filtration computed; exiting due to --filtration-only.\n";
        }

      // Boundary matrix computation + output
      if (!cfg.filtration_only && !cfg.output_stem.empty ())
        {
          if (cfg.verbose)
            std::cout << "Computing boundary matrix...\n";

          // No device memory used — inv_ordering (4 bytes/cell) is the only
          // intermediate host allocation beyond the phat representation.
          t0 = Clock::now ();
          auto result = plsf::compute_boundary_matrix (filtration);
          rows.push_back ({ "Boundary matrix", Ms (Clock::now () - t0).count (),
              cur_host_rss_kb (), -1 });

          if (cfg.pairs)
            {
              if (cfg.verbose)
                std::cout << "Computing persistence pairs...\n";

              t0 = Clock::now ();
              phat::persistence_pairs pairs;
              phat::compute_persistence_pairs (pairs, result.matrix);
              rows.push_back ({ "Persistence pairs",
                  Ms (Clock::now () - t0).count (), cur_host_rss_kb (), -1 });

              t0 = Clock::now ();
              pairs.save_binary (cfg.output_stem + ".pairs");
              plsf::write_filtration_values (
                  result.filt_values, cfg.output_stem);
              rows.push_back ({ "File I/O", Ms (Clock::now () - t0).count (),
                  cur_host_rss_kb (), -1 });

              if (cfg.verbose)
                std::cout << "Written : " << cfg.output_stem << ".pairs  "
                          << cfg.output_stem << ".vals\n";
            }
          else
            {
              t0 = Clock::now ();
              result.matrix.save_binary (cfg.output_stem + ".bin");
              plsf::write_filtration_values (
                  result.filt_values, cfg.output_stem);
              rows.push_back ({ "File I/O", Ms (Clock::now () - t0).count (),
                  cur_host_rss_kb (), -1 });

              if (cfg.verbose)
                std::cout << "Written : " << cfg.output_stem << ".bin  "
                          << cfg.output_stem << ".vals\n";
            }
        }
    }
  catch (const sycl::exception &e)
    {
      std::cerr << "SYCL error: " << e.what () << '\n';
      return 1;
    }
  catch (const std::exception &e)
    {
      std::cerr << "Error: " << e.what () << '\n';
      return 1;
    }

  if (cfg.timings)
    print_timings (rows);

  return 0;
}
