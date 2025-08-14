#!/usr/bin/env bash
set -euo pipefail

# --- Inputs ---
SEQ_TXT="training.seqs.txt"      # FASTA or 2-col: ID SEQ
LINKS="training.links.txt"       # Pairs: ID_A<ws>ID_B per line
SPLIT_FRACTION=0.9               # 90% clusters to train, 10% to val
THREADS="${THREADS:-$(nproc)}"
KEEP_TEMP="${KEEP_TEMP:-0}"      # KEEP_TEMP=1 to keep temp folder
SEED="${SEED:-}"                 # e.g., SEED=123 for reproducible split

# --- Outputs ---
TRAIN_SEQS="train.seqs.txt"      # ALWAYS: ID<TAB>SEQ, one line per sequence
VAL_SEQS="val.seqs.txt"
TRAIN_LINKS="train.links.txt"
VAL_LINKS="val.links.txt"

# --- Temp workspace ---
WORKDIR="./mint_split_tmp"
TMPDIR="$WORKDIR/mmseqs_tmp"
mkdir -p "$WORKDIR" "$TMPDIR"
cleanup(){ if [[ "$KEEP_TEMP" != "1" ]]; then rm -rf "$WORKDIR"; else echo "[i] Temp kept at $WORKDIR"; fi; }
trap cleanup EXIT

# 0) Normalize input -> one-line FASTA with headers truncated to FIRST TOKEN
SEQ_FASTA="$WORKDIR/training.seqs.fasta"
echo "[0/8] Normalize to one-line FASTA (headers -> first token only)"
if grep -q '^>' "$SEQ_TXT"; then
  awk '
    /^>/ {
      # flush previous seq
      if (NR>1 && seq!="") { print seq; seq="" }
      hdr=$0; sub(/^>/,"",hdr)
      split(hdr, h, /[ \t]+/)      # keep first token only
      print ">" h[1]
      next
    }
    { gsub(/\r/,""); gsub(/[ \t]/,""); seq = seq $0 }
    END { if (seq!="") print seq }
  ' "$SEQ_TXT" > "$SEQ_FASTA"
else
  awk '
    BEGIN{FS="[ \t]+"}
    /^[[:space:]]*$/ || /^#/ {next}
    {
      id=$1
      seq=$0
      sub("^[^ \t]+[ \t]+","",seq)
      gsub(/\r/,"",seq); gsub(/[ \t]/,"",seq)
      if (id=="" || seq=="") { print "Bad line (need ID and SEQ): " $0 > "/dev/stderr"; exit 1 }
      print ">" id "\n" seq
    }
  ' "$SEQ_TXT" > "$SEQ_FASTA"
fi

# 1) mmseqs createdb
echo "[1/8] mmseqs createdb"
mmseqs createdb "$SEQ_FASTA" "$WORKDIR/seqDB" >/dev/null

# 2) Cluster @ 40%
echo "[2/8] mmseqs linclust (0.4 id)"
mmseqs linclust "$WORKDIR/seqDB" "$WORKDIR/clu40" "$TMPDIR" \
  --min-seq-id 0.4 --cov-mode 1 -c 0.8 --threads "$THREADS" >/dev/null

# 3) Rep -> member map
echo "[3/8] mmseqs createtsv"
mmseqs createtsv "$WORKDIR/seqDB" "$WORKDIR/seqDB" "$WORKDIR/clu40" "$WORKDIR/clu40.tsv" >/dev/null
# format: <rep>\t<member> (IDs are first-token headers)

# 4) Split clusters
echo "[4/8] Split clusters"
cut -f1 "$WORKDIR/clu40.tsv" | sort -u > "$WORKDIR/clusters.all"
if [[ -n "$SEED" ]]; then
  shuf --random-source=<(printf "%s" "$SEED") "$WORKDIR/clusters.all" > "$WORKDIR/clusters.shuf"
else
  shuf "$WORKDIR/clusters.all" > "$WORKDIR/clusters.shuf"
fi
NUM_ALL=$(wc -l < "$WORKDIR/clusters.shuf")
if [[ "$NUM_ALL" -lt 2 ]]; then echo "Not enough clusters ($NUM_ALL) to split." >&2; exit 1; fi
NUM_TRAIN=$(python3 - << PY
import math
print(max(1, min($NUM_ALL-1, math.floor($NUM_ALL * $SPLIT_FRACTION))))
PY
)
head -n "$NUM_TRAIN" "$WORKDIR/clusters.shuf" > "$WORKDIR/clusters.train"
tail -n +"$((NUM_TRAIN+1))" "$WORKDIR/clusters.shuf" > "$WORKDIR/clusters.val" || true

# Expand to member IDs
awk 'NR==FNR{rep[$1]=1;next} ($1 in rep){print $2}' \
  "$WORKDIR/clusters.train" "$WORKDIR/clu40.tsv" | sort -u > "$WORKDIR/train.ids"
awk 'NR==FNR{rep[$1]=1;next} ($1 in rep){print $2}' \
  "$WORKDIR/clusters.val"   "$WORKDIR/clu40.tsv" | sort -u > "$WORKDIR/val.ids"

# 5) Ensure disjoint
echo "[5/8] Check disjoint ID sets"
if comm -12 <(sort "$WORKDIR/train.ids") <(sort "$WORKDIR/val.ids") | grep -q .; then
  echo "Error: Train/Val ID overlap detected." >&2; exit 1
fi

# 6) Emit 2-col ID<TAB>SEQ using a simple, streaming parser (matches mmseqs tokenization)
echo "[6/8] Emit train/val sequences (2-col ID\\tSEQ)"
emit_twocol() {
  local IDS="$1"
  local FASTA="$2"
  local OUT="$3"
  awk -v IDS_FILE="$IDS" '
    BEGIN{
      # read keep IDs
      while((getline l < IDS_FILE)>0){
        sub(/\r$/,"",l); gsub(/[ \t]+$/,"",l); gsub(/^[ \t]+/,"",l)
        keep[l]=1
      }
    }
    # parse FASTA streaming, no RS tricks
    /^>/ {
      # flush previous
      if (id!="" && keep[id]) print id "\t" seq
      seq=""
      header=$0; sub(/^>/,"",header)
      split(header, h, /[ \t]+/)   # first token only
      id=h[1]
      next
    }
    {
      gsub(/[ \t\r]/,"")
      seq=seq $0
    }
    END{
      if (id!="" && keep[id]) print id "\t" seq
    }
  ' "$FASTA" > "$OUT"
}

emit_twocol "$WORKDIR/train.ids" "$SEQ_FASTA" "$TRAIN_SEQS"
emit_twocol "$WORKDIR/val.ids"   "$SEQ_FASTA" "$VAL_SEQS"

# Sanity: counts must match IDs
TRAIN_COUNT=$(wc -l < "$TRAIN_SEQS")
VAL_COUNT=$(wc -l < "$VAL_SEQS")
TRAIN_ID_COUNT=$(wc -l < "$WORKDIR/train.ids")
VAL_ID_COUNT=$(wc -l < "$WORKDIR/val.ids")
if [[ "$TRAIN_COUNT" -ne "$TRAIN_ID_COUNT" || "$VAL_COUNT" -ne "$VAL_ID_COUNT" ]]; then
  echo "Error: sequence count != id count (train $TRAIN_COUNT vs $TRAIN_ID_COUNT, val $VAL_COUNT vs $VAL_ID_COUNT)." >&2
  echo "Debug hints:"
  echo "  head -3 $WORKDIR/train.ids"
  echo "  grep -m3 '^>' $SEQ_FASTA"
  exit 1
fi

# 7) Split links (intra-split only)
echo "[7/8] Split links"
awk 'NR==FNR{tr[$1]=1;next} (($1 in tr)&&($2 in tr))' "$WORKDIR/train.ids" "$LINKS" > "$TRAIN_LINKS"
awk 'NR==FNR{vl[$1]=1;next} (($1 in vl)&&($2 in vl))' "$WORKDIR/val.ids"   "$LINKS" > "$VAL_LINKS"

# 8) Stats
echo "[8/8] Counts:"
printf "  Train: %6d seqs, %6d links\n" "$TRAIN_COUNT" "$(wc -l < "$TRAIN_LINKS" || echo 0)"
printf "  Val  : %6d seqs, %6d links\n" "$VAL_COUNT"   "$(wc -l < "$VAL_LINKS"   || echo 0)"
echo "Done. Kept files:"
echo "  $TRAIN_SEQS, $VAL_SEQS, $TRAIN_LINKS, $VAL_LINKS"