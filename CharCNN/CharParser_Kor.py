CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"  # len = 19
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"  # len = 21
JONG = "ㄱ/ㄲ/ㄱㅅ/ㄴ/ㄴㅈ/ㄴㅎ/ㄷ/ㄹ/ㄹㄱ/ㄹㅁ/ㄹㅂ/ㄹㅅ/ㄹㅌ/ㄹㅍ/ㄹㅎ/ㅁ/ㅂ/ㅂㅅ/ㅅ/ㅆ/ㅇ/ㅈ/ㅊ/ㅋ/ㅌ/ㅍ/ㅎ".split('/')  # len = 27
HANGUL_LENGTH = len(CHO) + len(JUNG) + len(JONG)  # 67

GA_ASCII = ord('가')
HIH_ASCII = ord('힣')
G_ASCII = ord('ㄱ')
I_ASCII = ord('ㅣ')

def String_To_OneHot(str):
    one_hot_list = []
    for c in str:
        ascii = ord(c)
        one_hot = AsciiCode_To_OneHot(ascii)
        one_hot_list.extend(one_hot)
    
    return one_hot_list

def AsciiCode_To_OneHot(ascii):
    one_hot = []
    if GA_ASCII <= ascii <= HIH_ASCII:
        x = ascii - GA_ASCII
        y = x // 28
        z = x % 28
        x = y // 21
        y = y % 21

        one_hot.append(x)
        one_hot.append(len(CHO) + y)
        if z > 0:
            one_hot.append(len(CHO) + len(JUNG) + (z - 1))

    else:
        if ascii < 128:
            one_hot.append(HANGUL_LENGTH + ascii)  # 67~
        elif G_ASCII <= ascii <= I_ASCII:
            one_hot.append(HANGUL_LENGTH + 128 + (ascii - 12593))  # 194~ # [ㄱ:12593]~[ㅣ:12643] (len = 51)

    return one_hot