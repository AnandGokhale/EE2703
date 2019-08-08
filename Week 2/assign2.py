import numpy as np
import sys
import cmath
import math

CIRCUIT = ".circuit"
END = ".end"
AC = ".ac"

#from = lower voltage
#to = higher voltage

w = 0
ac_flag =0

class Node:
    def __init__(self,name,index):
        self.name = name
        self.index = index
        self.incurrent_passive = []
        self.outcurrent_passive = []
        self.incurrent_sourceV = []
        self.outcurrent_sourceV = []
        self.incurrent_sourceI = []
        self.outcurrent_sourceI = []
        self.voltage = None
        self.index = None

class Passive:
    def __init__(self,name,node1,node2,value,element):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        if(element == 'R'):
            self.value = value
        
        elif(element == 'C'):
            if(ac_flag):
                print("Do I work?s",w,value)
                self.value = complex(0,-1/(w*value))
                print(self.value)
            else:
                self.value = 1e100
        elif(element == 'L'):
            if(ac_flag):
                self.value = complex(0,(w*value))
            else:
                self.value = 1e-100
        self.element = element
    
       
class IndependentSources():
    def __init__(self,name,node1,node2,value,element):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value
        self.element = element




node =[]#list of Node Objects
nodes =[] #list of Node names
resistors = []
capacitors = []
inductors = []
voltage_sources = []
current_sources = []
nodes.append("GND")
dummy = Node("GND",0)
node.append(dummy)

#function reads file and returns the part between .circuit and .end
def fileread():
    global ac_flag,w
    if(len(sys.argv)!=2):
        print("Too Many Arguments")
        exit()
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        contains = []
        for l in lines:
            tokens = l.split()
            if(len(tokens) == 0):
                continue
            if (CIRCUIT== tokens[0]):
                flag = 1
                continue
            if flag:
                if (END== tokens[0] and (len(tokens)==1 or tokens[1][0] =='#')):
                    flag = 0
                contains.append(l)
            if(AC == tokens[0]  and tokens[1][0] == 'V'):
                
                ac_flag = 1
                try:
                    w = parse_val(tokens[2])
                except:
                    print("Missing Frquency")
                    exit()
                print("Frequency :" , w, ac_flag)
                w = w* 2*math.pi

                break
        if(len(contains)==0):
            print("Empty File or missing .circuit flag")
            exit()
    return contains

#converts string to float
def parse_val(x):
    y = len(x)
    if(not x[y-1].isalpha()):
        return float(x)
    if(x[y-1]=='p'):
        return float(x[0:y-1])* 1e-12   
    if(x[y-1]=='n'):
        return float(x[0:y-1])* 1e-9
    if(x[y-1]=='u'):
        return float(x[0:y-1])* 1e-6
    if(x[y-1]=='m'):
        return float(x[0:y-1])* 1e-3
    if(x[y-1]=='k'):
        return float(x[0:y-1])* 1e3
    if(x[y-1]=='M'):
        return float(x[0:y-1])* 1e6
    if(x[y-1]=='G'):
        return float(x[0:y-1])* 1e9  

def append(n1,n2):
    if(not (n1.isalnum() and n2.isalnum())):
        print("Node names are alphanumeric")
        exit()

    if(n1 not in nodes):
        nodes.append(n1)
        dummy = Node(n1,nodes.index(n1))
        node.append(dummy)
        
    if(n2 not in nodes):
        nodes.append(n2)
        dummy = Node(n2,nodes.index(n2))
        node.append(dummy)
    return nodes.index(n1),nodes.index(n2)


#function that breaks the line down into components, recognises the component and creates an object for the component
def parse_line(line):
    tokens = line.split()
    l = len(tokens)
    if(l==4 or (l>4 and tokens[4][0] =='#') and (tokens[0][0] == 'R' or tokens[0][0] == 'L' or tokens[0][0] == 'C' or tokens[0][0] == 'V' or tokens[0][0] == 'I')):
        element = tokens[0]
        n1 = tokens[1]
        n2 = tokens[2]
        value = tokens[3]
        val = parse_val(value)
        
        from_node_index,to_node_index = append(n1,n2)

        if(tokens[0][0] == 'R' or tokens[0][0] == 'C' or tokens[0][0] == 'L'):
            x = Passive(element,from_node_index,to_node_index,val,tokens[0][0])
            node[from_node_index].outcurrent_passive.append(x)
            
            node[to_node_index].incurrent_passive.append(x)
            if(tokens[0][0] == 'R'):
                resistors.append(x)            
            if(tokens[0][0] == 'L'):
                inductors.append(x)
            if(tokens[0][0] == 'C'):
                capacitors.append(x) 
        else:
            print("Syntax Error in netlist File")       
        
    elif(l == 6 or (l>6 and tokens[6][0] =='#')):
        if((tokens[0][0] == 'V' or tokens[0][0] == 'I') and tokens[3] == 'ac'):
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            value = parse_val(tokens[4])
            value/=2
            phase = parse_val(tokens[5])
            from_node_index,to_node_index = append(n1,n2)

            x = IndependentSources(element,from_node_index,to_node_index,complex(value*math.cos(phase),math.sin(phase)),tokens[0][0])
            if(tokens[0][0] == 'V'):
                voltage_sources.append(x)  
                node[from_node_index].outcurrent_sourceV.append(x)  
                node[to_node_index].incurrent_sourceV.append(x)        
            if(tokens[0][0] == 'I'):
                current_sources.append(x)
                node[from_node_index].outcurrent_sourceI.append(x)  
                node[to_node_index].incurrent_sourceI.append(x)  
        
        else:
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            n3 = tokens[3]
            n4 = tokens[4]
            value = tokens[5]

            if(not (n1.isalnum() and n2.isalnum() and n3.isalnum() and n4.isalnum())):
                print("Node names are alphanumeric")
                exit()

    elif(l==5 or (l>5 and tokens[5][0] =='#')):
        if((tokens[0][0] == 'V' or tokens[0][0] == 'I') and tokens[3] == 'dc'):
            if(ac_flag):
                print("Error:Multiple frequencies in same circuit")
                exit()
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            value = parse_val(tokens[4])
            from_node_index,to_node_index = append(n1,n2)

            if(tokens[0][0] == 'V'):
                x = IndependentSources(element,from_node_index,to_node_index,value,tokens[0][0])
                voltage_sources.append(x)  
                node[from_node_index].outcurrent_sourceV.append(x)  
                node[to_node_index].incurrent_sourceV.append(x)        
            if(tokens[0][0] == 'I'):
                x = IndependentSources(element,from_node_index,to_node_index,value,tokens[0][0])
                current_sources.append(x)
                node[from_node_index].outcurrent_sourceI.append(x)  
                node[to_node_index].incurrent_sourceI.append(x)  
            
        
        else:   
            element = tokens[0]
            n1 = tokens[1]
            n2 = tokens[2]
            V = tokens[3]
            value = tokens[4]
            if(not (n1.isalnum() and n2.isalnum())):
                print("Node names are alphanumeric")

    return

def populate_M():
    if(ac_flag==1):
        M = np.zeros((len(node)+len(voltage_sources),len(node)+len(voltage_sources)),dtype=np.complex)
        b = np.zeros(len(node)+ len(voltage_sources),dtype=np.complex)
    else:
        M = np.zeros((len(node)+len(voltage_sources),len(node)+len(voltage_sources)))
        b = np.zeros(len(node)+ len(voltage_sources))
    for n in node:
        #dealing with all resistors

        if(n.name == "GND"):
            M[0][0] = 1
            b[0] = 0
            continue
        for x in n.incurrent_passive:
            M[node.index(n)][x.node1] += (1/x.value)
            M[node.index(n)][node.index(n)] -= (1/x.value)
        for x in n.outcurrent_passive:
            M[node.index(n)][x.node2] += (1/x.value)
            M[node.index(n)][node.index(n)] -= (1/x.value)
    
    for x in voltage_sources:

        M[x.node1][voltage_sources.index(x)+len(node)] -=1
        M[x.node2][voltage_sources.index(x)+len(node)] +=1
        M[voltage_sources.index(x)+len(node)][x.node1] -=1    #from =  -ve
        M[voltage_sources.index(x)+len(node)][x.node2] +=1    #to = +ve
        b[voltage_sources.index(x)+len(node)] = x.value
        

    for x in current_sources:
        if(ac_flag==1):
            b[x.node1] += x.value/2                  #from = leaving
            b[x.node2] -= x.value/2                   #to = entering
        else:
            b[x.node1] += x.value                   #from = leaving
            b[x.node2] -= x.value
    M[0][len(node):] = np.zeros(len(voltage_sources))
    b[0] = 0
    return M,b


for l in fileread():
    print(l)
    parse_line(l)

M,b = populate_M()
print(M)
print(b)
try:
    X = np.linalg.solve(M,b)
except:
    print("Unsolvable Circuit")
    exit()

i=0
for n in nodes:
    print("Voltage at Node " + n+" =  " , cmath.polar(X[i]))
    i = i+1
for V in voltage_sources:
    print("Current Through Voltage Source " + V.name + " = ",cmath.polar(X[i]))
    i= i+1


#list of errors I Catch
'''
Unsolvable Matrix
Node names are alphanumeric
"Syntax Error in netlist File"
'''



 
