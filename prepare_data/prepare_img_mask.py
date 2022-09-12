
import os, glob
from PIL import Image
from google.cloud import storage
import shutil
import matplotlib.image as mpimg    
import matplotlib.pyplot as plt
import numpy as np



## List of the masks

## The location of the masks images

list_files=glob.glob('...\*.png')

print(len(list_files))



## In label studio if you tag an image a few times (want to correct previous image), it creates a few images that need to be concatenate

## creating list of images that has a few tags of the same image (only the 1/2/3 images)
list_not_0 = []
 
for file in list_files:
    if "0.png" not in file:
        list_not_0.append(file)
 

print (len(list_not_0))



## creating trans_image folders for each date folder

root='the folder with the tagged images'

for subdir, dirs, files in os.walk(root):
    for dir in dirs:
        path=os.path.join (root,dir,'tras_image')
        os.mkdir(path)




## converting the relevant files (not 0 files) to LA (add alpha for transpercy) for concatanating the images

def convertImage(file):
    
    img = Image.open(file)
    img = img.convert("LA")
  
    datas = img.getdata()
  
    newData = []
  
    for item in datas:
        if item[0]:
            newData.append((0, 0))
        else:
            newData.append(item)
  
    img.putdata(newData)
    img.save(os.path.dirname(file) + '/tras_image/'+ os.path.basename(file)+ "_New.png", "PNG")



for file in list_not_0:
   convertImage(file)
    



## choosing all the relevant files that need to be concatenate (o.png) 


tran_list= glob.glob ('/trans_image/*png')
file_to_merge=[]

for subdir, dirs, files in os.walk(root):
    for dir in dirs:
        if dir=="tras_image":
            path= subdir + '/' + dir + '/*png'
            tran_list=glob.glob (path)
            for task in tran_list:
                task_id=os.path.basename(task).split('-')[1]
                for file in list_files:
                    if task_id == os.path.basename(file).split('-')[1]:
                        if "0.png" in file:
                            file_to_merge.append(file)


print (len(file_to_merge))





## list of the converted to LA pic

tran_list= []

for subdir, dirs, files in os.walk(root):
    for dir in dirs:
        if dir=="tras_image":
            path= subdir + '/' + dir + '/*png'
            tran_list_d=glob.glob (path)
            tran_list.extend(tran_list_d)


## merge 0 and 1 masks

for file in file_to_merge:
    background= Image.open (file)
    task_id=os.path.basename(file).split('-')[1]
    for pic in tran_list:
        if task_id == os.path.basename(pic).split('-')[1]:
            if "1.png" in pic:
                path=os.path.dirname(pic)
                foreground=Image.open (pic)
                foreground.paste (background, (0, 0), background)
                foreground.save (os.path.dirname(pic) + "/task-" +task_id + "-merge_0_1.png", "PNG")
                
        




## take 2.png from the original file (without transparency) and concatenate to 0_1 pic

list_files_2=glob.glob('/*2.png')
merged=glob.glob('/*merge*')



for file in list_files_2:
    task_id=os.path.basename(file).split('-')[1]
    background=Image.open (file)
    for merge in merged:
                if task_id == os.path.basename(merge).split('-')[1]:
                    foreground= Image.open (merge)
                    foreground.paste (background, (0, 0), background)
                    foreground.save(os.path.dirname(merge) + "/task-" +task_id + "-merge_0_1_2.png", "PNG")
                                   
    





## take 3.png from the original file (without transparency)

list_files_3=glob.glob('/*3.png')


for file in list_files_3:
    task_id=os.path.basename(file).split('-')[1]
    background=Image.open (file)
    for merge in merged:
                if task_id == os.path.basename(merge).split('-')[1]:
                    foreground= Image.open (merge)
                    foreground.paste (background, (0, 0), background)
                    foreground.save (os.path.dirname(merge) + "/task-" +task_id + "-merge_0_1_2_3.png", "PNG")
                    
    




## checking if there are other numbers other than 1,2,3 in the labeled pic

greater_3=[]

for file in list_files:
    time=int(os.path.basename (file).split('-')[-1].split('.')[0])
    if time >3:
        greater_3.append(file)   
   
if len(greater_3)>0:
     print (greater_3)
else:
     print("No tagged files greater than 3")
            


print(greater_3)




## in case of 0_1_2_3, 0_1_2 and 0_1, deleting the 0_1 (in the concat_image)

## first creating list of relevant files to delete
merged=glob.glob(r'/tras_image/*merge*')

task_id_list_2=[]
task_id_list_3=[]

for file in merged:
    conc=os.path.basename(file).split('_')[-1]
    if conc=='2.png':
        task_id=os.path.basename(file).split('-')[1]
        #delet file with task_id and conc=1
        task_id_list_2.append(task_id)
        
print (task_id_list_2)

for file in merged:
    conc=os.path.basename(file).split('_')[-1]
    if conc=='3.png':
        task_id=os.path.basename(file).split('-')[1]
        #delet file with task_id and conc=1
        task_id_list_3.append(task_id)
        
print (task_id_list_3)



## creating folder to move the delete files to

path_delete='\\delete'
os.mkdir(path_delete)

delete=[]


## if there is a file with 2, delete the one
for task in task_id_list_2:
        for file in merged:
            num=os.path.basename(file).split('-')[1]
            if int(task) == int(num):
                if "1.png" in file:
                    delete.append (file)
                    
print(len(delete))

##if there is a file with 3, delete the 2
for task in task_id_list_3:
        for file in merged:
            num=os.path.basename(file).split('-')[1]
            if int(task) == int(num):
                if "2.png" in file:
                    delete.append (file)
                    
print (len(delete))

## "Removing this files to another folder
destination=r'D:\My Drive\tag_work\pic_biorad\label_studio\Meron_crop\delete'
for file in delete:
            os.rename (file, destination +'/' + os.path.basename(file) )




## move files of 0,1 and 2 from the source folders for putting instead the concatante files (with 0+1+2+3)

list_files=glob.glob('/*.png')

destination= r'\delete'

list_task_id=[]


## finding all the dubble picturs

for file in list_files:
    if "0.png" not in file:
        task_id=os.path.basename(file).split('-')[1]
        list_task_id.append(task_id)
        



## delete all the not merge files from the tras_image

files=glob.glob(r'\tras_image\*.png')

destination=r'\delete'



for file in files:
    if 'merge' not in file:
        os.rename (file,destination +'/' + os.path.basename(file) )



destination=r'\delete'

## remove the dubble picturs from the origin folders
for file in list_files:
    for task in list_task_id:
        if int(task)==int(os.path.basename(file).split('-')[1]):
            os.rename (file, destination +'/' + os.path.basename(file) )
            break



## adding the concat pic to the origin folder

root=r'\dates'

for subdir, dirs, files in os.walk(root):
    for dir in dirs:
      if dir=="tras_image":
        path= subdir + '/' + dir + '/*png'
       
        conc_list=glob.glob(path)
        for file in conc_list:
            os.rename (file, subdir +'/'+ os.path.basename(file))
            
   

## concat between the pic file name and the tag file name by their order
## Because the images from label-studio are without the name of orifin image

## sort both list

def num_tiff (file):
    num=int (os.path.basename (file.split('-')[3].split ('_')[0]))
    return num



def num_png (file):
    num=int (os.path.basename (file.split('-')[1]))
    return num





tiff_to_rename = glob.glob (r'\*.tiff')
png_to_rename = glob.glob (r'\*.png')



root='the folder with the tagged images'

for subdir, dirs, files in os.walk(root):
    for dir in dirs:
      if dir!="tras_image":
       path_png= subdir + '/' + dir + '/*png'
       path_tiff= subdir + '/' + dir + '/*tiff'
       png_to_rename=glob.glob(path_png)
       tiff_to_rename= glob.glob(path_tiff)
       if len (tiff_to_rename)!= len (png_to_rename): 
           print (dir, 'length is not equal')
          
       else:        
            for i, png in enumerate(sorted (png_to_rename, key= num_png)):
                    new_file_name = str(i)+'-' + os.path.basename(png)
                    os.rename(png, os.path.dirname(png) +'/'+ new_file_name)
                 
for subdir, dirs, files in os.walk(root):
    for dir in dirs:
      if dir!="tras_image":
       path_png= subdir + '/' + dir + '/*png'
       path_tiff= subdir + '/' + dir + '/*tiff'
   
       png_to_rename=glob.glob(path_png)
       tiff_to_rename= glob.glob(path_tiff)
       if len (tiff_to_rename)!= len (png_to_rename): 
           print (dir, 'length is not equal')
          
       else: 
        for i, tiff in enumerate(sorted(tiff_to_rename, key=num_tiff)):
                        new_file_name = str(i)+'-' + os.path.basename(tiff)
                        os.rename(tiff, os.path.dirname(tiff)+'/' + new_file_name) 
      

