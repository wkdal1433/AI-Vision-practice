youtuberlist = ["justlikethatkr", "paka", "ralo", "dopa"]



for index in youtuberlist:
    with open("{}.txt".format(index),'w',encoding="utf8") as email_file:
        contents = (f"안녕하세요? {index}님.\n\n"
        "(주)나도출판 편집자 나코입니다.\n"
        "현재 저희 출판사는 파이썬에 관한 주제로 책 출간을 기획 중입니다.\n"
        f"{index}님의 유튜브 영상을 보고 연락을 드리게 되었습니다.\n"
        "자세한 내용은 첨부드리는 제안서를 확인 부탁드리며, 긍정적인 회신 기다리겠습니다.\n\n"
        "좋은 하루 보내세요 ^^\n"
        "감사합니다.\n")
        email_file.write(contents)





    # f = open("C:/Users/wkdal/Desktop/python quiz/"+str(index)+".txt", 'w')
    # f.writelines('안녕하세요?? '+str(index)+'님.')
    # f.writelines('(주)나도출판 편집자 나코입니다.')
    # f.writelines('현재 저희 출판사는 파이썬에 관한 주제로 책 출간을 기획 중입니다.')
    # f.writelines(str(index)+'님의 유튜브 영상을 보고 연락을 드리게 되었습니다.')
    # f.writelines('자세한 내용은 첨부드리는 제안서를 확인 부탁드리며, 긍정적인 회신 기다리겠습니다.')
    # f.writelines()
    # f.writelines('좋은 하루 보내세요 ^^')
    # f.writelines('감사합니다.')

    # f.close()
    


