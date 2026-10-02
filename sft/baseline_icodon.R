# iCodon baseline (Diez et al., NAR 2022; github.com/santiago1234/iCodon).
#
# iCodon is third-party software and is not included in this repository: install
# the R package yourself (e.g. devtools::install_github("santiago1234/iCodon"))
# before running this script.
#
# iCodon optimizes PREDICTED mRNA STABILITY (decay rate) with a genetic algorithm
# over synonymous codons, and its predictor supports only human / mouse / fish /
# xenopus -- there is no fungal model. It is therefore run here as a human-host,
# stability-objective method, and belongs with GEMORNA / CodonBERT / codonGPT in
# the human-host group rather than with the fungal-host methods.
#
# The optimizer needs a starting CDS (it is a local search, not a
# protein-conditioned generator). Each target starts from its native CDS,
# adjusted where necessary so that it encodes the panel protein exactly
# (see sft/data/icodon_start_cds.json). Several random seeds are run per target
# so the method gets a candidate set instead of a single point.

suppressMessages(library(iCodon))

args <- commandArgs(trailingOnly = TRUE)
in_fasta  <- args[1]
out_fasta <- args[2]
n_seeds   <- as.integer(args[3])
specie    <- if (length(args) >= 4) args[4] else "human"
n_iter    <- if (length(args) >= 5) as.integer(args[5]) else 15

read_fasta <- function(path) {
  lines <- readLines(path)
  names_idx <- grep("^>", lines)
  out <- list()
  for (i in seq_along(names_idx)) {
    start <- names_idx[i] + 1
    end <- if (i < length(names_idx)) names_idx[i + 1] - 1 else length(lines)
    out[[sub("^>", "", lines[names_idx[i]])]] <- paste(lines[start:end], collapse = "")
  }
  out
}

seqs <- read_fasta(in_fasta)
con <- file(out_fasta, "w")
for (target in names(seqs)) {
  start_seq <- seqs[[target]]
  for (s in seq_len(n_seeds)) {
    res <- optimizer(start_seq, specie = specie, n_iterations = n_iter,
                     make_more_optimal = TRUE, random_seed = s)
    # optimizer returns the best sequence at each iteration; take the last row
    best <- res[[which(sapply(res, is.character))[1]]]
    best <- best[length(best)]
    n_codon <- nchar(best) %/% 3
    writeLines(sprintf(">icodon|%s|%d n_codon=%d", target, s - 1, n_codon), con)
    writeLines(best, con)
    cat(sprintf("%s seed %d: %d nt\n", target, s, nchar(best)))
    flush.console()
  }
}
close(con)
cat(sprintf("wrote %s\n", out_fasta))
