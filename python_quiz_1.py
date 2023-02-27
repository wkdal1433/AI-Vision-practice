youtuberlist = ["justlikethatkr", "paka", "ralo", "dopa"]



for index in youtuberlist:
    # youtubername = youtuberlist(index)
    f = open("C:/Users/wkdal/Desktop/python quiz/"+str(index)+".txt", 'w')
    f.write('안녕하세요? '+str(index)+'님.\n(주)나도출판 편집자 나코입니다.\n현재 저희 출판사는 파이썬에 관한 주제로 책 출간을 기획 중입니다. \n'+str(index)+'님의 유튜브 영상을 보고 연락을 드리게 되었습니다. \n자세한 내용은 첨부드리는 제안서를 확인 부탁드리며, 긍정적인 회신 기다리겠습니다. \n\n좋은 하루 보내세요 ^^ \n감사합니다.')
    f.close()
    


