library("optparse")
#library("ExpressionSet")
library("Biobase")
library("TDARACNE")
 
option_list = list(
    make_option(c("--cond"),           type="character",   default="epid1", help="epid1 | epid2 | neur | endo | meso", metavar="character"),
    make_option(c("--indir"),          type = "character", default="data", help="directory with cluster data"),
    make_option(c("--gtpmf"),          type = "character", default="gexp/counts_consold/animal_cap_tpm",  help="csv with all tpm data"),
    make_option(c("--gnmf"),           type = "character", default="gexp/BMP_gnms",  help="csv where the first column is genes that will be networked"),
    make_option(c("--adjf"),           type = "character", default="adj_mat",  help="csv with all tpm data"),
    make_option(c("--outdir"),         type = "character", default="data", help="directory to write deseq data"),
    make_option(c("--idxCode"),        type = "character", default="111111", help="replicate indexes to use per stage, unless useMean=TRUE"),

    make_option(c("--quest"),          type = "logical", default=FALSE, help="whether running on quest"),
    make_option(c("--useMean"),        type = "logical", default=FALSE, help="whether to average all replicates; ignores idxCode"),
    make_option(c("--appendMetaInfo"), type = "logical", default=FALSE, help="whether to add metainfo onto the output filename"),

    # tdaracne options
    make_option(c("--nbins"),        type = "numeric", default=11,    help="# bins used for normalization"),
    make_option(c("--delta"),        type = "numeric", default=5,     help="max time delay allowed to infer connections"),
    make_option(c("--likelihood"),   type = "numeric", default=1.2,   help="fold change used as threshold to state the initial change expression"),
    make_option(c("--rank_norm"),    type = "logical", default=FALSE, help="true = row normalization, false = rank normalization"),
    make_option(c("--logged_data"),  type = "logical", default=FALSE, help="true if input matrix is log transformed"),
    make_option(c("--thresh"),       type = "numeric", default=0,     help="Influence threshold...."),
    make_option(c("--ksd"),          type = "numeric", default=1,     help="standard deviation multiplier"),
    make_option(c("--tol"),          type = "numeric", default=0.15,  help="DPI (Data Processing Inequality) tolerance. 0 means no tolerance and 1 means no DPI") 

); 

opt_parser = OptionParser(option_list=option_list);
opt        = parse_args(opt_parser);

quest_head = "/projects/p31095/simonf/out/xenopus"
home_head  = "/Users/simonfreedman/cqub/xenopus"
headdir    = if (opt$quest) quest_head else home_head

indir  = paste(headdir, "/", opt$indir,  sep="");
outdir = paste(headdir, "/", opt$outdir, sep="");

gnmf  = paste(indir,"/",  opt$gnmf,".csv",sep="")
gtpmf = paste(indir,"/",  opt$gtpmf,".csv",sep="")
adjf  = paste(outdir,"/", opt$adjf,".csv",sep="")

stgs    = c("9","10","10.5","11","12","13")

gtpm       = read.table(gtpmf, header=TRUE, sep=",")
cond_mat   = read.table(text = colnames(gtpm), sep="_")

tmat       = matrix(0, dim(gtpm)[1], length(stgs))
#tsdmat     = matrix(0, dim(gtpm)[1], len(stgs))
stg_idxs   = strtoi(unlist(strsplit(opt$idxCode,"")))
print(cond_mat)
for (i in 1:6){
    cond_stg_idxs = which(cond_mat[,1]==opt$cond & cond_mat[,2]==paste("stg",stgs[i],sep=""));
    print(cond_stg_idxs)
    if (opt$useMean){
        tmat[,i]      = rowMeans(gtpm[,cond_stg_idxs])
#        tsdmat[,i]    = rowSds(gtpm[,cond_stg_idxs])
    }else{
        tmat[,i]   = gtpm[,cond_stg_idxs[stg_idxs[i]]]
    }
}
rownames(tmat)   = rownames(gtpm)
#rownames(tsdmat) = rownames(gtpm)

#fnm2='/Users/simonfreedman/cqub/xenopus/data/gexp/counts_consold/animal_cap_tpm.csv'

#bmp_fnm = '/Users/simonfreedman/cqub/xenopus/data/gexp/BMP_gnms.csv'
gnms    = read.table(gnmf, header=TRUE, sep=",")[,1]
exprSet = ExpressionSet(tmat[gnms,])

rank_norm   = if (opt$rank_norm) 2 else 1
logged_data = if (opt$logged_data) 0 else 1
#thresh      = if (opt$useMean) tstd_mat else opt$thresh

adjmat  = TDARACNE(exprSet,N=opt$nbins, delta=opt$delta, likehood = opt$likelihood, 
                   norm = rank_norm, logarithm = logged_data, thresh = opt$thresh, ksd = opt$ksd, tolerance = opt$tol, adj=TRUE);

write.table(adjmat, adjf, row.names=TRUE,col.names=TRUE,sep=',')
