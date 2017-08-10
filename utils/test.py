import re

def isac(c):
    try:
        int(c)
        return True
    except:
        if c == '.' or c == '-' or c == 'e':
            return True
        else:
            return False

def find_all_serial(string_phase):
    str_tmp = ""
    for i in range(len(string_phase)):
        if not isac(string_phase[i]):
            str_tmp += ' '
        else:
            str_tmp += string_phase[i]
    print(str_tmp)

if __name__ == '__main__':
    phs = ": 0.91e-12"
    find_all_serial(phs)