import sys
import string

def str2tokens(line):
    tokens = line.split()
    l = len(tokens)
    if(l==4 or (l>4 and tokens[4][0] =='#')):
        element = tokens[0]
        n1 = tokens[1]
        n2 = tokens[2]
        value = tokens[3]
        if(not (n1.isalnum() and n2.isalnum())):
            print("Node names are alphanumeric")
            return
        print(value,n2,n1,element)
    elif(l == 6 or (l>6 and tokens[6][0] =='#')):
        element = tokens[0]
        n1 = tokens[1]
        n2 = tokens[2]
        n3 = tokens[3]
        n4 = tokens[4]
        value = tokens[5]

        if(not (n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum())):
            print("Node names are alphanumeric")
            return
        print(value,n4,n3,n2,n1,element)
    elif(l==5 or (l>5 and tokens[5][0] =='#')):
        element = tokens[0]
        n1 = tokens[1]
        n2 = tokens[2]
        V = tokens[3]
        value = tokens[4]
        if(not (n1.isalnum() and n2.isalnum())):
            print("Node names are alphanumeric")
            return
        print(value,V,n2,n1,element)
    return

try:
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        flag = 0
        contains = []
        for l in reversed(lines):
            tokens = l.split()
            if (".end" == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#')):
                flag = 1
                continue
            if flag:
                if (".circuit" == tokens[0] and (len(tokens)==1 or tokens[1][0] =='#')):
                    break
                contains.append(l)
                try:
                    str2tokens(l)
                except :
                    print("Invalid Format!!!")
except :
    print("File Not Found!!")