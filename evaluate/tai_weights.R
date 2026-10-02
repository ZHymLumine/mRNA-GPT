# Computes the 60-codon relative-adaptiveness (w) vector for the tRNA
# Adaptation Index (dos Reis, Savva & Wernisch 2004), using the exact
# reference implementation (https://github.com/mariodosreis/tai, R/tAI.R),
# fed with S. cerevisiae tRNA gene copy numbers from GtRNAdb
# (evaluate/data/scer_trna_gene_counts.json).
#
# Usage: Rscript tai_weights.R <tai_source.R> <trna_json> <out_json>
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
source(args[1])                              # defines get.ws()
trna <- fromJSON(args[2])

ws <- get.ws(tRNA = trna$trna_gene_count, sking = 0)   # 0 = Eukaryota
stopifnot(length(ws) == 60)

# get.ws drops positions 11,12,15 (stop) and 36 (Met) from the 64-codon order,
# so the surviving 60 codons are codon_order with those 4 removed, in order.
codon_order_60 <- trna$codon_order[-c(11, 12, 15, 36)]

write(toJSON(list(codon_order = codon_order_60, w = ws), auto_unbox = FALSE, digits = 10),
      args[3])
cat("wrote", args[3], "\n")
