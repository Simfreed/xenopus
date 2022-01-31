library("DESeq2")
library("optparse")
 
option_list = list(
    make_option(c("--fnm1"), type="character", default=NULL, help="cluster 1", metavar="character"),
    make_option(c("--fnm2"), type="character", default=NULL, help="cluster 2", metavar="character"),
    make_option(c("--sampfrac"), type="numeric", default=1, help="fraction of data to use"),
    make_option(c("--indir"), type = "character", default="single_cell_raw", help="directory with cluster data"),
    make_option(c("--outdir"), type = "character", default="deseq_results", help="directory to write deseq data"),
    make_option(c("--seed"), type = "integer", default=1, help="random number seed for sampling"),
    make_option(c("--replace"), type = "logical", default=FALSE, help="whether to sample with replacement"),
    make_option(c("--appendMetaInfo"), type = "logical", default=FALSE, help="whether to add metainfo onto the output filename"),
    make_option(c("--dispersions"), type = "logical", default=FALSE, help="whether to write neg. binom. dispersions to file"),
    make_option(c("--nsamp"), type = "integer", default=-1, help="alternative to sampfrac if set>=0. Enforces equal number of samples for both conditions."),
    make_option(c("--nhdlines"), type="numeric", default=9, help="number of header lines")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
##args=(commandArgs(TRUE))
#setwd("~/cqub/xenopus/data/")


# should be nsamps1 and nsamps2 nbased on samp frac and indices should be randomized

gnmf = paste(opt$indir,"/row_names.tsv",sep="")
f1   = paste(opt$indir,"/", opt$fnm1, ".tsv", sep="")
f2   = paste(opt$indir,"/", opt$fnm2, ".tsv", sep="")

nhdlines = opt$nhdlines 
gnms     = read.table(gnmf, skip = nhdlines, stringsAsFactors = FALSE, sep="\t")[,"V1"]
paste("read gene names")

t1 = read.table(f1, skip = nhdlines, row.names = gnms)
paste("read t1")
t2 = read.table(f2, skip = nhdlines, row.names = gnms)
paste("read t2")

nc1 = dim(t1)[2]
nc2 = dim(t2)[2]

nsamp1 = floor(opt$sampfrac * nc1)
nsamp2 = floor(opt$sampfrac * nc2)

if (opt$nsamp >= 0){
    nsamp  = min(nc1, nc2, opt$nsamp)
    nsamp1 = nsamp
    nsamp2 = nsamp
}

paste("using", nsamp1, "samples of", opt$fnm1, "and", nsamp2, "samples of", opt$fnm2)

set.seed(opt$seed)
samp1idxs = sample(1:nc1, nsamp1, replace=opt$replace)
samp2idxs = sample(1:nc2, nsamp2, replace=opt$replace)

samp1 = c(1:nsamp1)
samp2 = c(1:nsamp2)

cond1 = array(opt$fnm1, dim=nsamp1)
cond2 = array(opt$fnm2, dim=nsamp2)

exp_matrix = data.frame(c(samp1,samp2),c(cond1,cond2))
colnames(exp_matrix) = c("sample", "condition")

#cdat = cbind(t1,t2)
cdat = cbind(t1[,samp1idxs],t2[,samp2idxs])

start.time <- Sys.time()


dds_matrix = DESeqDataSetFromMatrix(countData=cdat, colData=exp_matrix, design = ~condition)
dds_test   = DESeq(dds_matrix, sfType = "poscounts", fitType = "local")
# alternatively, 
# sfType = "iterate" -- but seems to take a lot longer (default = "ratio" doesn't work because of too many 0s) 
# fitType = "mean" -- but not default alternative behavior 
# (default = "parametric" doesn't capture dispersion trend well according to DESeq error messages, so the default alternative is "local")
dds_res        = results(dds_test)
dds_res_sorted = dds_res #dds_res[order(dds_res$padj),]
dds_disps      = dispersions(dds_test)
end.time <- Sys.time()

paste("DESeq execution time: ")
end.time-start.time

nmAppend = ""
if (opt$appendMetaInfo){
    nmAppend = paste("_n1",nsamp1,"_n2",nsamp2,"_seed",opt$seed,sep="")
}
if (opt$dispersions){
    write.table(as.data.frame(dds_disps), file=paste(opt$outdir,"/",opt$fnm1,"_v_",opt$fnm2,nmAppend,"_dispersions.tsv",sep=""), sep="\t")
}
write.table(as.data.frame(dds_res_sorted), file=paste(opt$outdir,"/",opt$fnm1,"_v_",opt$fnm2,nmAppend,".tsv",sep=""), sep="\t")

