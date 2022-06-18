#ASSIGNMENT-1

def gradient_function(x):
    g=2*(x-1)
    return g
#learing rate 
learning_rate = 0.0001 
#initial x
x = 10 

i = 0  
for i in range(1000000) :
    previous_x = x
    x = x - learning_rate * gradient_function(x)
    print(x)
    i += 1
    #if x < the minimum value of negative value - or -the grident ha no change in its value
    if x < 1e-6 or x - previous_x==0:
        break
    

  