from yolo_dataset import xml_to_namelist, readxml_to_df, make_fd_train_test

source_folder = 'my_data_images' # ชื่อโฟลเดอร์ ต้นทาง ที่เก็บรูป jpg และ ไฟล์ xml
for_train_folder = 'my_data_images_for_train' # ชื่อโฟลเดอร์ ปลายทาง ที่จะใช้เก็บรูป jpg และ ไฟล์ txt

object_name = {'Symbols_address':0, 'Symbols_UDI':1, 'Symbols_TH':2, 'Symbols_temp':3, 'Symbols_Humi':4} # ชื่อ Object ทั้ง 3 ของเรา

print('Number of Class : ' + str(len(object_name))) # แสดงจำนวนชื่อ Object

xml_img_path = './' + source_folder + '/*.xml'
xmlfiles = xml_to_namelist(xml_img_path)
print('Number of Images : ' + str(len(xmlfiles))) # แสดงจำนวนภาพทั้งหมด

df = readxml_to_df(xmlfiles)
print(df.head(10))

ratio_train = 0.8 # แบ่ง 80% สำหรับ Train และ 20% สำหรับ Valid
make_fd_train_test(df,ratio_train,object_name,source_folder,for_train_folder)


