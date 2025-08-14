#!/bin/bash

# Script to download files from RCSB http file download services.
# Use the -h switch to get help on usage.

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "Loading DATA_DIR from the .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Check for curl or wget
if command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
elif command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
else
    echo "Neither 'curl' nor 'wget' could be found. You need to install either one for this script to work."
    exit 1
fi

echo "Using $DOWNLOAD_CMD for downloads"

PROGNAME=$0
BASE_URL="https://files.rcsb.org/download"

usage() {
  cat << EOF >&2
Usage: $PROGNAME [-f <file>] [-o <BASE_DATA_DIR>] [-n <cpus>] [-c] [-p] [-a] [-A] [-x] [-s] [-m] [-r]

 -o  <BASE_DATA_DIR>: the base data directory (optional, defaults to DATA_DIR from .env file)
 -f  <file>: Overrides the input file containing a comma-separated list of PDB ids
             (default: BASE_DATA_DIR/raw/pdb_ids.txt)
 -n  <cpus>: number of CPUs to use for parallel downloads, default: 16
 -c       : download a cif.gz file for each PDB id
 -p       : download a pdb.gz file for each PDB id (not available for large structures)
 -a       : download a pdb1.gz file (1st bioassembly) for each PDB id (not available for large structures)
 -A       : download an assembly1.cif.gz file (1st bioassembly) for each PDB id
 -x       : download a xml.gz file for each PDB id
 -s       : download a sf.cif.gz file for each PDB id (diffraction only)
 -m       : download a mr.gz file for each PDB id (NMR only)
 -r       : download a mr.str.gz for each PDB id (NMR only)
EOF
  exit 1
}

download() {
    url="$BASE_URL/$1"
    out="$2/$1"
    
    # Check if the file already exists
    if [ -f "$out" ]; then
        echo "File $out already exists. Skipping download."
        return
    fi

    echo "Downloading $url to $out"
    if [ "$DOWNLOAD_CMD" = "curl" ]; then
        curl -s -f "$url" -o "$out" || echo "Failed to download $url"
    else
        wget -q "$url" -O "$out" || echo "Failed to download $url"
    fi
}

listfile=""
base_dir=""
outdir=""
num_cpus=16
cif=false
pdb=false
pdb1=false
cifassembly1=false
xml=false
sf=false
mr=false
mrstr=false

while getopts f:o:n:cpaAxsmr o
do
  case $o in
    (f) listfile=$OPTARG;;
    (o) base_dir=$OPTARG;;
    (n) num_cpus=$OPTARG;;
    (c) cif=true;;
    (p) pdb=true;;
    (a) pdb1=true;;
    (A) cifassembly1=true;;
    (x) xml=true;;
    (s) sf=true;;
    (m) mr=true;;
    (r) mrstr=true;;
    (*) usage
  esac
done
shift "$((OPTIND - 1))"

# If no base directory provided, try to use DATA_DIR from environment
if [ -z "$base_dir" ]; then
    if [ -n "$DATA_DIR" ]; then
        base_dir="$DATA_DIR"
        echo "Using DATA_DIR from the .env file: $base_dir"
    else
        echo "Error: Parameter -o (BASE_DATA_DIR) must be provided or DATA_DIR must be set in .env file"
        usage
        exit 1
    fi
fi

# Set output directory
outdir="$base_dir/raw/cif_zipped"

# If no listfile specified, use default path
if [ -z "$listfile" ]; then
    listfile="$base_dir/raw/pdb_ids.txt"
fi

# Check if listfile exists
if [ ! -f "$listfile" ]; then
    echo "Error: Input file not found: $listfile"
    exit 1
fi

# Create cif_zipped directory within the output directory
mkdir -p "$outdir"
echo "Files will be downloaded to: $outdir"
echo "Using input file: $listfile"

contents=$(cat "$listfile")

# Split comma-separated list of tokens into an array
IFS=',' read -ra tokens <<< "$contents"

# Echo the number of CPUs being used
echo "Using $num_cpus CPU(s) for parallel downloads."

# Function to wait until the number of background jobs is less than the CPU count.
wait_for_jobs() {
  while [ "$(jobs -r | wc -l)" -ge "$num_cpus" ]; do
    sleep 0.1
  done
}

# For each PDB id token, launch the selected downloads in the background,
# waiting when necessary to limit the concurrency.
for token in "${tokens[@]}"
do
  if [ "$cif" == true ]
  then
    wait_for_jobs
    download "${token}.cif.gz" "$outdir" &
  fi
  if [ "$pdb" == true ]
  then
    wait_for_jobs
    download "${token}.pdb.gz" "$outdir" &
  fi
  if [ "$pdb1" == true ]
  then
    wait_for_jobs
    download "${token}.pdb1.gz" "$outdir" &
  fi
  if [ "$cifassembly1" == true ]
  then
    wait_for_jobs
    download "${token}-assembly1.cif.gz" "$outdir" &
  fi
  if [ "$xml" == true ]
  then
    wait_for_jobs
    download "${token}.xml.gz" "$outdir" &
  fi
  if [ "$sf" == true ]
  then
    wait_for_jobs
    download "${token}-sf.cif.gz" "$outdir" &
  fi
  if [ "$mr" == true ]
  then
    wait_for_jobs
    download "${token}.mr.gz" "$outdir" &
  fi
  if [ "$mrstr" == true ]
  then
    wait_for_jobs
    download "${token}_mr.str.gz" "$outdir" &
  fi
done

# Wait for all background downloads to complete before exiting.
wait