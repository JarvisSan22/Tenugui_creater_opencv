import cv2
import numpy as np
import matplotlib.pyplot as plt


#画像の解析##################################################
def RBGhist(img):
  h,w,c=img.shape
  #RBGのヒストグラム
  fig,axs=plt.subplots(2,1,figsize=(8,8))
  #monoのimgsを作成する
  RBG_imgs=[]
  colors=["blue","green","red"]
  for coli,col in zip(range(c),colors):
    #monoの画像
    img_c=np.zeros((h,w,c),dtype=np.uint8)
    img_c[:,:,coli]=img[:,:,coli]
    RBG_imgs.append(img_c)
    #色によりヒストグラム
    hist_c=cv2.calcHist([img],[coli],None,[256],[1,255])
    axs[1].plot(hist_c,color=col,label=col)

  RBG_monos=cv2.hconcat(RBG_imgs)
  axs[0].imshow(cv2.cvtColor(RBG_monos,cv2.COLOR_BGR2RGB))
  axs[0].set_axis_off()
  axs[1].legend()
  plt.show()

def grayhist(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_gray_hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
  fig,axs=plt.subplots(2,1,figsize=(8,8))
  axs[0].imshow(img_gray,cmap="gray")
  axs[0].set_axis_off()
  axs[1].plot(img_gray_hist,color="gray")

#画像の処理
def Threshold(img,min_limit=None, max_limit=256,plot=False):
    _, binary = cv2.threshold (img, min_limit,max_limit, cv2.THRESH_BINARY)
    b_max=np.max(binary)
    if b_max>=255:
      mask=binary[:,:,:]==[b_max,b_max,b_max]
      binary[np.logical_or.reduce(mask,axis=2)]=[255,255,255]

    if plot:
        plt.imshow(binary, cmap="gray")
    return binary

def Smoothing(img,kernal,blurtype="blur"):
    if blurtype.lower()=="blur":
        img_blur = cv2.blur(img, kernal)
    elif blurtype.lower()=="gaussianblur":
        img_blur = cv2.GaussianBlur(img,kernal, 1)
    return img_blur

#画像の変身##################################################
def RoateImage(img,angle,plot=False):
  h,w,c=img.shape
  
  #rotation matrix 
  side=(h-w)/2
  M1=np.float32([[1,0,side],[0,1,0]])
  img_tr1= cv2.warpAffine(img,M1,(h,h))
  #中央店で反時計真璃々に角度で回転
  M2=cv2.getRotationMatrix2D((w/2+side,h/2),angle,1)
  #出力
  img_rot=cv2.warpAffine(img_tr1,M2,(h,h))
  
  #上に余分に移動する -side
#  if h!=w:
#    M3=np.float32([[1,0,0],[0,1,-side]])
#    img_rot=cv2.warpAffine(img_rot,M3,(h,w))

  if plot:
    plt.imshow(np.hstack([img,img_rot]))
    print(h,w)
    h,w,c=img_rot.shape
    print(h,w)
  return img_rot

def grayscale_3channel(img):
  img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h,w=img_gray.shape
  gray_3C=np.zeros([h,w,3])
  for i in [0,1,2]:
    gray_3C[:,:,i]=img_gray
  return gray_3C


#模様を作成する##################################################
#ゼロのArreyに加える方法
def CmtoPx(cm):
  #https://www.unitconverters.net/typography/centimeter-to-pixel-x.htm
  return int(cm*37.7952755906)
#ゼロのArreyに加える方法
def CmtoPx(cm):
  #https://www.unitconverters.net/typography/centimeter-to-pixel-x.htm
  return int(cm*37.7952755906)
def CreatePatten(img,hcm,wcm,padcm,buff_w=20,buff_h=20,rotation=True,angle=None,plot=True):
  
  bhcm=hcm+padcm #cm #66
  bwcm=wcm+padcm #cm #36

  base_array=np.zeros((CmtoPx(bhcm),CmtoPx(bwcm),3),dtype=np.uint8)
  print(base_array.shape)
  bh,bw,c=base_array.shape
  h,w,c=img.shape

  h_n=bh//w
  w_n=bw//h
  #print(h_n,bh)
  #print(w_n,bw)
  
  img_cut_h=h
  img_cut_w=w
  #最初の位置
  wp=0
  hp=0
  
  #画像の間に間隔
  #buff_w=buff_w
  #buff_h=buff_h
  
  img_pl=img.copy()
  rotangle=angle
  for i in range(h_n):
    for j in range(w_n):
      if rotation:
         img_pl=RoateImage(img.copy(),rotangle)
         rotangle+=angle
         if rotangle >360:
          rotangle=rotangle-360
      try:
        #print(rotangle)
        if i % 2 ==0:   
          base_array[hp:hp+h,wp+w//2:wp+w+w//2,:]=img_pl[:,:,:]  
        else:
          base_array[hp:hp+h,wp:wp+w,:]=img_pl[:,:,:]  
        rotangle+=angle

       

      except Exception as e:
       # print(e)
       # print(img_pl.shape)
        pass
      wp+=w+buff_w
    wp=0
    hp+=h+buff_h



  #Cut to 手ぬぐい
  thpx=CmtoPx(hcm)
  twpx=CmtoPx(wcm)
  croped_array=base_array[(bh-thpx)//2:bh-(bh-thpx)//2,(bw-twpx)//2:bw-(bw-twpx)//2,:]
  if plot==True:
    plt.figure(figsize=(10,15))
    plt.imshow(croped_array)
  return croped_array

def colorpatten(pattern,black_color,white_color,plot=True):
  colored_patten=pattern.copy()
  maskc1= colored_patten==[0,0,0]
  colored_patten[np.logical_or.reduce(maskc1,axis=2)]=white_color
  maskc2=pattern[:,:,:]==[255,255,255]
  colored_patten[np.logical_or.reduce(maskc2,axis=2)]=black_color
  if plot==True:
    plt.figure(figsize=(10,15))
    plt.imshow( colored_patten)
    plt.axis('off')
  return colored_patten

def main():
  test_file="./test_pic/rain_dragon.jpg"
  img=cv2.imread(test_file)
  img_gray=grayscale_3channel(img)

  #grayhist(img)
  min_limit=100
  max_limit=255
  img_thresh=Threshold(img_gray,min_limit=min_limit, max_limit=max_limit,plot=False)
  img_resize=cv2.resize(img_thresh,(300,300))

  #模様を作成する
  hcm=100 #cm #66
  wcm=60 #cm #36
  padcm=0
  angle=55
  Pattern=CreatePatten(img_resize,hcm,wcm,padcm,rotation=True,angle=angle)
  #色を付ける
  c2=[20,42,90]
  c1=[183,233,244]
  Colored_Patten=colorpatten(Pattern,c1,c2)

  #cut
  nhcm=66
  nwcm=36
  print(CmtoPx(nhcm),CmtoPx(nwcm))
  cuth=CmtoPx((hcm-nhcm)//2)
  cutw=CmtoPx((wcm-nwcm)//2)
  h,w,c=Colored_Patten.shape
  Colored_Patten=Colored_Patten[cuth:h-cuth,cutw:w-cutw,:]
  Colored_Patten=cv2.cvtColor(Colored_Patten, cv2.COLOR_BGR2RGB)

  Colored_Patten=cv2.resize(Colored_Patten,(CmtoPx(nwcm),CmtoPx(nhcm)))
  print(Colored_Patten.shape)
  cv2.imwrite("./test_pic/Tenugui_raindragon.png",Colored_Patten)

if __name__=="__main__":
  main()

