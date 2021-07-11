from google_images_download import google_images_download 
import os
response = google_images_download.googleimagesdownload()

keyword = "cat"

#num이 100개 이상 시 selenium을 사용
#사용 시 크롬드라이버 설정 필요
def downloadimages(query):
    num=300
    arguments = {"keywords": query,        #다운하고 싶은 이미지 키워드
                 "format": "jpg",             #다운로드 확장자
                 "limit": num,                #다운 이미지 개수
                 "output_directory": "image",
                 "chromedriver": r"C:/Users/Owner/Desktop/인공지능_기말/chromedriver.exe"} #'image'라는 폴더를 만들어서 그 안에 저장
                 
    try:
        response.download(arguments)
    
    # Handling File NotFound Error
    except FileNotFoundError:  
        arguments = {"keywords": query, 
                     "format": "jpg", 
                     "limit":num,
                     "output_directory": "image",
                     "chromedriver": r"C:/Users/Owner/Desktop/인공지능_기말/chromedriver.exe"} 
                    
        try: 
            response.download(arguments)  
        except: 
            pass

#downloadimages(keyword) #이미지 다운
#print()   #다운 받은 이미지 출력


# 파일 이름 변경
file_path = './image/dog'
file_names = os.listdir(file_path)

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = 'dog' + str(i) + '_.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
