"""Command-line interface for physics data fetching and processing.

Usage:
    python -m smatrix_2d.physics_data.cli fetch --all
    python -m smatrix_2d.physics_data.cli fetch --source nist_pstar --material H2O
    python -m smatrix_2d.physics_data.cli generate-lut --type scattering --material H2O
    python -m smatrix_2d.physics_data.cli info
"""

import argparse
import logging
from pathlib import Path

from smatrix_2d.physics_data.fetchers import fetch_nist_pstar, fetch_pdg_constants
from smatrix_2d.physics_data.fetchers.nist_pstar import get_nist_materials
from smatrix_2d.physics_data.processors import (
    process_stopping_power,
    generate_scattering_lut_from_raw,
)
from smatrix_2d.physics_data.processors.scattering import ScatteringModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default paths - data is stored within the package at smatrix_2d/data/
PACKAGE_ROOT = Path(__file__).parent.parent  # smatrix_2d/
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch raw physics data from external sources."""
    logger.info("Fetching physics data...")

    if args.all or args.source == "nist_pstar":
        logger.info("Fetching NIST PSTAR data...")

        if args.material == "all":
            materials = get_nist_materials()
        else:
            materials = [args.material]

        for material in materials:
            try:
                data = fetch_nist_pstar(
                    material=material,
                    output_dir=RAW_DIR,
                    use_fallback=True,
                )
                logger.info(
                    f"Fetched {len(data.energy)} points for {material}: "
                    f"{data.energy[0]:.3f} - {data.energy[-1]:.1f} MeV"
                )
            except Exception as e:
                logger.error(f"Failed to fetch {material}: {e}")

    if args.all or args.source == "pdg":
        logger.info("Fetching PDG constants...")
        constants = fetch_pdg_constants(output_dir=RAW_DIR)
        logger.info(f"Fetched PDG constants (version {constants.pdg_version})")

    logger.info(f"Data saved to {RAW_DIR}")


def cmd_generate_lut(args: argparse.Namespace) -> None:
    """Generate lookup tables from raw data."""
    logger.info("Generating lookup tables...")

    if args.type == "stopping_power" or args.type == "all":
        logger.info(f"Generating stopping power LUT for {args.material}...")

        lut = process_stopping_power(
            material=args.material,
            raw_data_path=RAW_DIR / f"nist_pstar_{args.material.lower()}.csv",
            E_min=args.energy_min,
            E_max=args.energy_max,
            n_points=args.energy_points,
            grid_type=args.grid_type,
            output_path=PROCESSED_DIR / f"stopping_power_{args.material.lower()}.npy",
        )

        logger.info(
            f"Generated stopping power LUT: {lut.energy_grid[0]:.3f} - "
            f"{lut.energy_grid[-1]:.1f} MeV ({len(lut.energy_grid)} points)"
        )

    if args.type == "scattering" or args.type == "all":
        logger.info(f"Generating scattering LUT for {args.material}...")

        model = ScatteringModel.MOLIERE if args.model == "moliere" else ScatteringModel.HIGHLAND

        lut = generate_scattering_lut_from_raw(
            material=args.material,
            E_min=args.energy_min,
            E_max=args.energy_max,
            n_points=args.energy_points,
            grid_type=args.grid_type,
            model=model,
            output_path=PROCESSED_DIR / f"scattering_{args.material.lower()}.npy",
        )

        logger.info(
            f"Generated scattering LUT: {lut.energy_grid[0]:.3f} - "
            f"{lut.energy_grid[-1]:.1f} MeV ({len(lut.energy_grid)} points)"
        )

    logger.info(f"LUTs saved to {PROCESSED_DIR}")


def cmd_info(args: argparse.Namespace) -> None:
    """Display information about available data and sources."""
    print("\n" + "=" * 60)
    print("PHYSICS DATA INFORMATION")
    print("=" * 60)

    print("\n[Data Sources]")
    print(f"  NIST PSTAR: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html")
    print(f"  PDG:        https://pdg.lbl.gov/")

    print("\n[Available Materials (NIST PSTAR)]")
    for i, mat in enumerate(get_nist_materials()[:10], 1):
        print(f"  {i:2d}. {mat}")
    print(f"  ... and {len(get_nist_materials()) - 10} more")

    print("\n[Data Directories]")
    print(f"  Raw data:      {RAW_DIR}")
    print(f"  Processed LUTs: {PROCESSED_DIR}")
    print(f"  Metadata:      {METADATA_DIR}")

    # Check what data exists
    print("\n[Existing Data]")
    raw_files = list(RAW_DIR.glob("*.*")) if RAW_DIR.exists() else []
    processed_files = list(PROCESSED_DIR.glob("*.*")) if PROCESSED_DIR.exists() else []

    print(f"  Raw files:      {len([f for f in raw_files if not f.name.startswith('.')])}")
    for f in raw_files[:5]:
        if not f.name.startswith("."):
            print(f"    - {f.name}")
    if len(raw_files) > 5:
        print(f"    ... and {len(raw_files) - 5} more")

    print(f"  Processed files: {len([f for f in processed_files if not f.name.startswith('.')])}")
    for f in processed_files[:5]:
        if not f.name.startswith("."):
            print(f"    - {f.name}")
    if len(processed_files) > 5:
        print(f"    ... and {len(processed_files) - 5} more")

    print("\n" + "=" * 60)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Physics data fetching and processing for Smatrix 2D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all raw data
  python -m smatrix_2d.physics_data.cli fetch --all

  # Fetch NIST data for water
  python -m smatrix_2d.physics_data.cli fetch --source nist_pstar --material H2O

  # Generate stopping power LUT
  python -m smatrix_2d.physics_data.cli generate-lut --type stopping_power --material H2O

  # Generate scattering LUT with custom energy grid
  python -m smatrix_2d.physics_data.cli generate-lut --type scattering --material H2O \\
      --energy-min 1.0 --energy-max 250.0 --energy-points 200 --grid-type logarithmic

  # Show information
  python -m smatrix_2d.physics_data.cli info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch raw data from external sources")
    fetch_parser.add_argument(
        "--source",
        choices=["nist_pstar", "pdg", "all"],
        default="all",
        help="Data source to fetch (default: all)",
    )
    fetch_parser.add_argument(
        "--material",
        default="all",
        help="Material to fetch (default: all)",
    )
    fetch_parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all sources",
    )

    # Generate LUT command
    lut_parser = subparsers.add_parser("generate-lut", help="Generate lookup tables")
    lut_parser.add_argument(
        "--type",
        choices=["stopping_power", "scattering", "all"],
        default="all",
        help="Type of LUT to generate (default: all)",
    )
    lut_parser.add_argument(
        "--material",
        default="H2O",
        help="Material (default: H2O)",
    )
    lut_parser.add_argument(
        "--energy-min",
        type=float,
        default=1.0,
        help="Minimum energy [MeV] (default: 1.0)",
    )
    lut_parser.add_argument(
        "--energy-max",
        type=float,
        default=250.0,
        help="Maximum energy [MeV] (default: 250.0)",
    )
    lut_parser.add_argument(
        "--energy-points",
        type=int,
        default=200,
        help="Number of energy points (default: 200)",
    )
    lut_parser.add_argument(
        "--grid-type",
        choices=["uniform", "logarithmic"],
        default="logarithmic",
        help="Energy grid type (default: logarithmic)",
    )
    lut_parser.add_argument(
        "--model",
        choices=["moliere", "highland"],
        default="moliere",
        help="Scattering model (default: moliere)",
    )

    # Info command
    subparsers.add_parser("info", help="Display information about available data")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "generate-lut":
        cmd_generate_lut(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
