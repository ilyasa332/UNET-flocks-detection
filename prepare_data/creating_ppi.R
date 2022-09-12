library(bioRad)
library(stringr)
library(viridis)



## list of relevant H5 files

list_files<- list.files("location_of_h5_files",recursive = TRUE, full.names = TRUE)


plot_list_v=list()

for (file in list_files){
  pvol<-read_pvolfile(file)
  scan<-pvol$scans[[1]]
  ppi<-project_as_ppi(scan)
  
  
 
  p<-plot(ppi,  param="VRADH",frame.plot=F)+
    viridis::scale_fill_viridis(name = "VRADH")+
    theme(panel.background = element_rect(fill = "black")) +
    theme(panel.border = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = "none",
          axis.title = element_blank(),
          axis.text = element_blank(),
          axis.ticks = element_blank())+
    geom_text(x=0, y=-53000,colour="White" ,label=ppi$datetime)
  
  
  
  plot_list_v[[file]]=p
}

# Save plots to tiff

## The year, month and day are relavant to files in a format of:1316ISR-PPIVol-20180910-150216-286e.hdf


for (i in list_files) {
  year=substr(basename(i),16,19)
  month=substr(basename(i),20,21)
  day=substr(basename(i),22,23)
  file_name = paste("D:/",year,'/',month,'/',day,'/',str_remove_all( str_sub(basename(i),end=-10),"[A-z]"),"_VRADH", ".tiff", sep="")
  tiff(file_name)
  print(plot_list_v[[i]])
  dev.off()
}

