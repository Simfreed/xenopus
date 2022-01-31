library("DESeq2")
library("optparse")
 
option_list = list(
    make_option(c("--nm1"), type="character", default=NULL, help="1st counts file name", metavar="character"),
    make_option(c("--nm2"), type="character", default=NULL, help="2nd counts file name", metavar="character"),
    make_option(c("--gnmf"), type="character", default="row_names", help="gene names file", metavar="character"),
    make_option(c("--useHead"), type="logical", default=FALSE, action="store_true",
                help="file should be {nm1}_head.csv and {nm2}_head.csv"),
    make_option(c("--indir"), type = "character", default="counts", help="directory with cluster data"),
    make_option(c("--outdir"), type = "character", default="deseq_results", help="directory to write deseq data"),
    make_option(c("--appendMetaInfo"), type = "logical", default=FALSE, help="whether to add metainfo onto the output filename"),
    make_option(c("--nhdlines"), type="numeric", default=0, help="number of header lines")
); 
 
opt_parser = OptionParser(option_list=option_list);
opt        = parse_args(opt_parser);

# should be nsamps1 and nsamps2 nbased on samp frac and indices should be randomized

gnmf = paste(opt$indir,"/", opt$gnmf,".tsv",sep="")
f1   = paste(opt$indir,"/", opt$nm1, ".tsv", sep="")
f2   = paste(opt$indir,"/", opt$nm2, ".tsv", sep="")

nhdlines = opt$nhdlines 
gnms     = read.table(gnmf, skip = nhdlines, stringsAsFactors = FALSE, sep="\t")[,"V1"]

t1 = read.table(f1, skip = nhdlines, row.names = gnms, sep="\t")
paste("read t1")
t2 = read.table(f2, skip = nhdlines, row.names = gnms)
paste("read t2")

nsamp1 = dim(t1)[2]
nsamp2 = dim(t2)[2]

paste("using", nsamp1, "samples from", opt$nm1, "and", nsamp2, "samples from", opt$nm2)

cond1 = array(opt$nm1, dim=nsamp1)
cond2 = array(opt$nm2, dim=nsamp2)

if(opt$useHead){
    print("reading header files")
    h1 = read.csv(paste(opt$indir,"/", opt$nm1, "_head.csv", sep=""), header=FALSE)
    h2 = read.csv(paste(opt$indir,"/", opt$nm2, "_head.csv", sep=""), header=FALSE)
    samps1 = as.character(as.vector(h1[1,]))
    times1 = as.character(as.vector(h1[2,]))
    samps2 = as.character(as.vector(h2[1,]))
    times2 = as.character(as.vector(h2[2,]))
}else{
    print("using default header")
    times = as.character(as.vector(t(cbind(1:6,1:6,1:6))))
    samps = as.character(as.vector(cbind(1:3,1:3,1:3,1:3,1:3,1:3)))
    times1 = times
    samps1 = samps
    times2 = times
    samps2 = samps
}

exp_matrix = data.frame(c(cond1,cond2),c(times1,times2),c(samps1,samps2))
colnames(exp_matrix) = c("condition", "stage","sample")
cdat = cbind(t1,t2)

start.time <- Sys.time()

#ddsTC <- DESeqDataSet(fission, ~ strain + minute + strain:minute)
dds_matrix = DESeqDataSetFromMatrix(countData=cdat, colData=exp_matrix, design = ~ condition + stage + condition:stage)

dds_test   = DESeq(dds_matrix, test="LRT", reduced = ~ condition + stage)

# dds_test   = DESeq(dds_matrix, sfType = "poscounts", fitType = "local")
# alternatively, 
# sfType = "iterate" -- but seems to take a lot longer (default = "ratio" doesn't work because of too many 0s) 
# fitType = "mean" -- but not default alternative behavior 
# (default = "parametric" doesn't capture dispersion trend well according to DESeq error messages, so the default alternative is "local")
dds_res        = results(dds_test)
dds_res_sorted = dds_res #dds_res[order(dds_res$padj),]

end.time <- Sys.time()

paste("DESeq execution time: ")
end.time-start.time

nmAppend = ""
if (opt$appendMetaInfo){
    nmAppend = paste("_n1",nsamp1,"_n2",nsamp2,sep="")
}

write.table(as.data.frame(dds_res_sorted), file=paste(opt$outdir,"/",opt$nm1,"_v_",opt$nm2,nmAppend,".tsv",sep=""), sep="\t")

