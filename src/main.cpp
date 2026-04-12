// main.cpp
// ─────────────────────────────────────────────────────────────────────────────
// gLSF – GPU Lower-Star Filtration
// CLI entry point: parse arguments → load volume → run filtration → save output
// ─────────────────────────────────────────────────────────────────────────────

#include "filtration/cubical_grid.hpp"
#include "filtration/lower_star.hpp"
#include "io/volume_io.hpp"
#include "sycl_utils.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

// ── CLI configuration
// ─────────────────────────────────────────────────────────
struct Config
{
  std::string input_path;
  std::string output_path = "pairs.csv";
  std::string device_hint = "gpu"; // "cpu" | "gpu" | "default"
  float       threshold   = 0.0f;  // minimum persistence to report
  bool        binary_out  = false; // write binary instead of CSV
  bool        verbose     = false;

  // Raw binary overrides (used when input has no header)
  bool        raw_mode = false;
  std::size_t raw_nx = 0, raw_ny = 0, raw_nz = 0;
  std::size_t raw_offset = 0;
};

static void print_usage(const char* prog)
{
  std::cout
      << "Usage: " << prog
      << " [options] <input>\n"
         "\n"
         "Options:\n"
         "  -o, --output  <file>      Output path for persistence pairs "
         "[pairs.csv]\n"
         "  -d, --device  <mode>      SYCL device selector: cpu|gpu|default  "
         "[gpu]\n"
         "  -t, --threshold <val>     Minimum persistence (death-birth)     "
         "[0]\n"
         "  -b, --binary              Write pairs as binary instead of CSV  "
         "[off]\n"
         "  -v, --verbose             Print device info and cell statistics  "
         "[off]\n"
         "  -h, --help                Show this message and exit\n"
         "\n"
         "Raw binary input (no header):\n"
         "  --raw <nx> <ny> <nz>      Treat <input> as headerless float32   \n"
         "  --raw-offset <bytes>      Bytes to skip at start of raw file    "
         "[0]\n"
         "\n"
         "Supported auto-detected formats: .nii  .nii.gz  .mhd\n";
}

static Config parse_args(int argc, char* argv[])
{
  if (argc < 2)
  {
    print_usage(argv[0]);
    std::exit(1);
  }

  Config cfg;
  for (int i = 1; i < argc; ++i)
  {
    std::string a = argv[i];
    if (a == "-h" || a == "--help")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    else if (a == "-b" || a == "--binary")
      cfg.binary_out = true;
    else if (a == "-v" || a == "--verbose")
      cfg.verbose = true;
    else if ((a == "-o" || a == "--output") && i + 1 < argc)
      cfg.output_path = argv[++i];
    else if ((a == "-d" || a == "--device") && i + 1 < argc)
      cfg.device_hint = argv[++i];
    else if ((a == "-t" || a == "--threshold") && i + 1 < argc)
      cfg.threshold = std::stof(argv[++i]);
    else if (a == "--raw" && i + 3 < argc)
    {
      cfg.raw_mode = true;
      cfg.raw_nx   = static_cast<std::size_t>(std::stoull(argv[++i]));
      cfg.raw_ny   = static_cast<std::size_t>(std::stoull(argv[++i]));
      cfg.raw_nz   = static_cast<std::size_t>(std::stoull(argv[++i]));
    }
    else if (a == "--raw-offset" && i + 1 < argc)
      cfg.raw_offset = static_cast<std::size_t>(std::stoull(argv[++i]));
    else if (a[0] != '-')
      cfg.input_path = a;
    else
    {
      std::cerr << "Unknown option: " << a << '\n';
      std::exit(1);
    }
  }

  if (cfg.input_path.empty())
  {
    std::cerr << "Error: no input file specified.\n";
    std::exit(1);
  }
  return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  const Config cfg = parse_args(argc, argv);

  // ── Select SYCL device ─────────────────────────────────────────────────────
  glsf::DevicePreference dev_pref = glsf::DevicePreference::GPU;
  if (cfg.device_hint == "cpu")
    dev_pref = glsf::DevicePreference::CPU;
  else if (cfg.device_hint == "default")
    dev_pref = glsf::DevicePreference::Default;

  sycl::queue queue = glsf::make_queue(dev_pref);
  if (cfg.verbose)
    glsf::print_device_info(queue);

  try
  {
    // ── Load volume ──────────────────────────────────────────────────────────
    if (cfg.verbose)
      std::cout << "Loading: " << cfg.input_path << '\n';

    glsf::CubicalGrid3D<float> grid;
    glsf::io::VolumeMetadata   meta;

    if (cfg.raw_mode)
    {
      std::tie(grid, meta) = glsf::io::read_raw<float>(
          cfg.input_path, cfg.raw_nx, cfg.raw_ny, cfg.raw_nz, cfg.raw_offset);
    }
    else
    {
      std::tie(grid, meta) = glsf::io::read_volume<float>(cfg.input_path);
    }

    if (cfg.verbose)
    {
      std::cout << "Grid    : " << meta.dims[0] << " × " << meta.dims[1]
                << " × " << meta.dims[2] << "  (" << grid.num_cells()
                << " cells total)\n"
                << "Spacing : " << meta.spacing[0] << " × " << meta.spacing[1]
                << " × " << meta.spacing[2] << " mm\n";
    }

    // ── Run filtration ────────────────────────────────────────────────────
    glsf::LowerStarFiltration<float> filtration;
    filtration.compute(queue, grid);

    auto pairs = filtration.significant_pairs(cfg.threshold);

    if (cfg.verbose)
      std::cout << "Pairs   : " << pairs.size() << "  (threshold "
                << cfg.threshold << ")\n";

    // ── Write output ──────────────────────────────────────────────────────
    if (cfg.binary_out)
      glsf::io::write_pairs_bin(cfg.output_path, pairs);
    else
      glsf::io::write_pairs_csv(cfg.output_path, pairs);

    if (cfg.verbose)
      std::cout << "Written : " << cfg.output_path << '\n';
  }
  catch (const sycl::exception& e)
  {
    std::cerr << "SYCL error: " << e.what() << '\n';
    return 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
