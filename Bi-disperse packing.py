#import relevant modules
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math
import random


#Generate points on a circle
def circle(c, r):
    #c, r - centre and radius
    th = np.linspace(0, 2 * np.pi, 100)
    xunit = r * np.cos(th) + c[0]
    yunit = r * np.sin(th) + c[1]
    values=np.stack((xunit,yunit),axis=1)
    return values

#Calculate position of next disc
def solver(s1, s2, s3, r1, r2):
    #s1, r1 and s2, r2 - radii and positions of previous discs. s3 - radius of new disc 
    u=r1-r2
    mag_u=np.sqrt(np.dot(u,u))

    cb=((s1+s3)**2-(s3+s2)**2-(mag_u)**2)/(-2*mag_u*(s3+s2))
    sb=np.sqrt(1-cb**2)

    beta=math.acos(cb)

    ct=u[0]/mag_u
   
    #required for anticlockwise positioning - take +ve
    if abs(u[1])==0:
        st=0
    else:
        st=(u[1]/abs(u[1]))*np.sqrt(1-ct**2)

    theta_u=math.acos(u[0]/mag_u)
   
    rho=(s2+s3)*(np.array([ct*cb+st*sb,st*cb-ct*sb]))
    new_pos=r2+rho
   

    return new_pos

#Calculate area of cell
def area(indices, positions):
    A=0
    for i in range(0, len(indices)-2):
        x=positions[indices[i+2]]-positions[indices[i+1]]
        y=positions[indices[i+1]]-positions[indices[0]]
        
        A+=(0.5*(x[0]*y[1]-x[1]*y[0]))
        
    return round(abs(A), 5)

     
#Calculate angle between two vectors
def angle(v1, v2):
    if -v1[0]*v2[1]+v1[1]*v2[0] == 0:
        return 0
    else:
        sign=(-v1[0]*v2[1]+v1[1]*v2[0])/abs(v1[0]*v2[1]-v1[1]*v2[0])
        return(math.acos(np.dot(v1,v2)/(math.sqrt(np.dot(v1,v1)*np.dot(v2,v2))))*sign)
    
def cell_angle(indices, positions):
    angles=[]
    for i in range(len(indices)):
        theta=round(np.pi+angle(np.round(positions[indices[(i+1)%(len(indices))]]-positions[indices[i]],decimals=5), np.round(positions[indices[i]]-positions[indices[i-1]],decimals=5)),3)
        angles.append(theta)
        
    return angles

#checks for an overlap for newly placed disc
def overlap(new_pos, new_D, root, prev_index, positions, sizes, forw, back):
        
    over=False
    
    #forward
    for i in range(100): #checking ahead by 100
        
        if root+i == prev_index-back:
            break
    
        elif root+i==root+forw:
            pass
        
        
        else:
            disp=round((((positions[root+i][0]-new_pos[0])**2 + (positions[root+i][1]-new_pos[1])**2)**(0.5)), 5)
            if disp < new_D+sizes[root+i]:
                over=True
                break
            
            else:
                pass
        
    #backward
    for i in range(100): #checking backwards by 100
        
        #print('checking backwards')
        
        if prev_index-i==prev_index-back:
            pass
        
        elif prev_index-i==root+forw:
            pass
         
        elif prev_index-i<0:
            pass
        
        else:
            disp=round(((positions[prev_index-i][0]-new_pos[0])**2 + (positions[prev_index-i][1]-new_pos[1])**2)**(0.5),5)
            if disp < new_D+sizes[prev_index-i]:
                over=True
                break
            
            else:
                pass
     
    return over

#checks for an overlap for newly placed disc
def exact_overlap(new_pos, new_D, root, prev_index, positions, sizes):
        
    
    forward=[]
    
    #forward
    for i in range(1,4): #only checking 4 ahead
        
        if root+i == prev_index:
            break

        
        else:
            disp=round((((positions[root+i][0]-new_pos[0])**2 + (positions[root+i][1]-new_pos[1])**2)**(0.5)), 5)
            if disp == new_D+sizes[root+i]:
                forward.append(root+i)
            
            else:
                pass
    backward=[]    
    
    #backward
    for i in range(1,4): #only checking 4 behind
        
        if prev_index-i<0:
            pass
        
        elif prev_index-i==root:
            pass
        
        else:
            disp=round(((positions[prev_index-i][0]-new_pos[0])**2 + (positions[prev_index-i][1]-new_pos[1])**2)**(0.5),5)
            if disp == new_D+sizes[prev_index-i]:
                backward.append(prev_index-i)
            else:
                pass
    
    return forward, backward #returns of the exact overlap sites



#Defining parameters
D=3.70 #disc ratio
P=0.5 #concentration of discs of size 1
N=1000 #number of discs 

num_forward=50 #number to check beyond the root site
num_backward=50 #number to check behind the previously placed site


#Initial setup


sizes=np.array([1,1,D]) #disc diameters
positions=np.array([[0,0],[2,0],[1,np.sqrt(D**2+2*D)]]) #disc positions
pre=[1,D,1] #set disordered nucleus


cell_distribution=np.array([3]) #cell orders
cell_areas=np.array([area(np.array([0,1,2]),positions)]) #cell areas
root=0 #lowest available site
chain=np.array([0,1,2]) #longest unbroken chain
total_disc_area=((sizes[0]**2+sizes[1]**2+sizes[2]**2)*np.pi) #total area of discs
cell_angles=np.array(cell_angle([1,0,2], positions)) #cell area distribution


conditional_area=[[3, area(np.array([0,1,2]),positions)]] #conditional area distribution
conditional_angle=[[3, cell_angle([1,0,2], positions)]] #conditional angle distribution

cells=[[1,0,2]] #cell indices

crystal_count=0 #number of crystals (3-cells of same D) in a row


#Generates the lattice
for x in range(N-3):
    order=3
    
    #previously placed disc values
    prev_pos = positions[-1]
    prev_D = sizes[-1] 
    prev_index = len(sizes)-1

    if x>=len(pre): #to allow for the initially disordered nucleus
        if random.random() < P: #chooses D randomly with a given concentration
            new_D = 1 

        else:
            new_D = D

    else:
        new_D=pre[x]


    #new_pos=solver(sizes[root],prev_D,new_D,positions[root],prev_pos) #initial trial for the new position

    back=0 #number backwards for previous site

    overlapping=True #while a new position has not been found

    while overlapping == True:

        i=0 #number trialed beyond the root site

        while i <= num_forward: 

            if root+i >= prev_index-back: #avoid checking beyond the one contact
                i+=1
                break

            elif round((((positions[root+i][0]-positions[prev_index-back][0])**2 + (positions[root+i][1]-positions[prev_index-back][1])**2)**(0.5)),5) <= (2*new_D+sizes[root+i]+sizes[prev_index-back]): #check that placement is possible

                new_pos=solver(sizes[root+i],sizes[prev_index-back],new_D,positions[root+i],positions[prev_index-back]) #trial site for disc

                status=overlap(new_pos, new_D, root, prev_index, positions, sizes, i, back) #check for overlap

                if status==False and (i+back)==0:#if no overlap and have a 3-cell

                    if sizes[root+i]==new_D and sizes[prev_index-back]==new_D: #check for crystallisation
                        crystal_count+=1
                        if sizes[root+i+1]==new_D and round((((positions[root+i+1][0]-new_pos[0])**2 + (positions[root+i+1][1]-new_pos[1])**2)**(0.5)),5) == (2*new_D):
                            #additional exact overlap, adding an extra crystal
                            crystal_count+=1
                            
                        else:
                            pass
                    
                    else:
                        pass

                    if crystal_count>2: #if have more than two crystals in a row
                        crystal_count=0
                        i=0
                        #force the other value of D
                        if new_D==1:
                            new_D=D
                        else:
                            new_D=D

                        new_pos=solver(sizes[root],prev_D,new_D,positions[root],prev_pos)

                        continue #restart the loop

                    else:
                        root+=i #update the root
                        overlapping=False #position has been found
                        break


                elif status==False: #no overlap but cell order is greater than 3
                    crystal_count=0 #no crystallisation

                    root+=i #update root
                    overlapping=False #position has been found
                    break

                else: #try the next root site
                    i+=1

            else:
                i+=1 #try the next root site (if possible)

        back+=1 #try updating the previous index site

    positions=np.append(positions,[new_pos],axis=0) #add the position
    sizes=np.append(sizes,[new_D])  #add the size

    exact=exact_overlap(new_pos, new_D, root, prev_index-back+1, positions, sizes) #check for an exact overlap
    exact[0].insert(0, root) #insert root 
    exact[1].insert(0, prev_index-(back-1)) #previous site

    cell_indices=np.concatenate((chain[np.where(chain == (prev_index)-(back-1))[0][0]:np.where(chain == (prev_index))[0][0]+1],chain[np.where(chain == root-i)[0][0]:np.where(chain == root)[0][0]+1],[prev_index+1])) 
    cells.append(cell_indices) #add the cell indices

    cell_areas=np.append(cell_areas, area(cell_indices, positions)) #add the area
    cell_distribution=np.append(cell_distribution, len(cell_indices)) #add the distribution
    cell_angles=np.append(cell_angles, cell_angle(cell_indices, positions)) #add the angle

    conditional_area.append([len(cell_indices),area(cell_indices, positions)]) #update the conditional area distribution
    conditional_angle.append([len(cell_indices), cell_angle(cell_indices, positions)]) #update the conditional angle distribution

    if len(exact[0])>1: #if there is an exact overlap forwards

        for v in range(len(exact[0])-1):
            cell_indices_add=np.concatenate((chain[np.where(chain == exact[0][v])[0][0]:np.where(chain == exact[0][v+1])[0][0]+1],[prev_index+1]))
            #additional cell properties
            cells.append(cell_indices_add) 
            cell_areas=np.append(cell_areas, area(cell_indices_add, positions))
            cell_distribution=np.append(cell_distribution, len(cell_indices_add))
            cell_angles=np.append(cell_angles, cell_angle(cell_indices_add, positions))
            conditional_area.append([len(cell_indices_add),area(cell_indices_add, positions)])
            conditional_angle.append([len(cell_indices), cell_angle(cell_indices, positions)])
            
    else:
        pass

    take=0

    if len(exact[1])>1: #if there is an exact overlap backwards
        take+=prev_index-(back-1)-exact[1][-1]
        for v in range(0, len(exact[1])-1):
            cell_indices_add=np.concatenate((chain[np.where(chain == exact[1][v+1])[0][0]:np.where(chain == exact[1][v])[0][0]+1],[prev_index+1]))
            #additional cell properties
            print(cell_indices_add) 
            cells.append(cell_indices_add)
            cell_areas=np.append(cell_areas, area(cell_indices_add, positions))
            cell_angles=np.append(cell_angles, cell_angle(cell_indices_add, positions))
            cell_distribution=np.append(cell_distribution, len(cell_indices))
            conditional_angle.append([len(cell_indices), cell_angle(cell_indices, positions)])

    else:
        pass



    chain=chain[0:(np.where(chain == prev_index-(back-1)-take)[0][0]+1)] #update the longest chain
    chain=np.append(chain, x+3)
    total_disc_area+=(np.pi*(new_D)**2)
    temp=total_disc_area


    root=exact[0][-1]


    if x<-1: #visualisation during packing

        for center, radius in zip(positions, sizes):
            x = circle(center, radius)
            plt.plot(x[:,0],x[:,1], 'black', linewidth=0.1)

        for coor in cells:
            x=[]
            y=[]
            for i in coor:
                x.append(positions[i][0])
                y.append(positions[i][1])

            x.append(positions[coor[0]][0])
            y.append(positions[coor[0]][1])
            plt.plot(x,y)

            plt.plot(x,y,'g')


        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')    
        plt.show()

        for center, radius in zip(positions, sizes):
            x = circle(center, radius)
            plt.plot(x[:,0],x[:,1], 'black', linewidth=0.1)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')    
        plt.show()

        largest_polygon=chain[np.where(chain==root)[0][0]:len(chain)]

        for i in range(len(largest_polygon)):
            theta=round(np.pi+angle(np.round(positions[largest_polygon[(i+1)%(len(largest_polygon))]]-positions[largest_polygon[i]],decimals=5), np.round(positions[largest_polygon[i]]-positions[largest_polygon[i-1]],decimals=5)),3)
            total_disc_area-=(theta)*(sizes[largest_polygon[i]]**2)*0.5

        total_area=(area(largest_polygon, positions))

        packing_fraction=(total_disc_area/total_area)
        print('Packing fraction: '+str(packing_fraction))
        
        total_disc_area=temp

        
largest_polygon=chain[np.where(chain==root)[0][0]:len(chain)]

if D != 1:
    label_p, count_p =np.unique(sizes, return_counts=True)
    print('Actual p: '+str(count_p[0]/(count_p[0]+count_p[1])))
    total_area_large=np.pi*(count_p[1]*label_p[1]**2)
    total_area_small=np.pi*(count_p[0]*label_p[0]**2)
    
else:
    pass


for i in range(len(largest_polygon)):
    theta=round(np.pi+angle(np.round(positions[largest_polygon[(i+1)%(len(largest_polygon))]]-positions[largest_polygon[i]],decimals=5), np.round(positions[largest_polygon[i]]-positions[largest_polygon[i-1]],decimals=5)),3)
    total_disc_area-=(theta)*(sizes[largest_polygon[i]]**2)*0.5
    if sizes[largest_polygon[i]] == label_p[0] and D != 1:
        total_area_small-=(theta)*(sizes[largest_polygon[i]]**2)*0.5
    elif sizes[largest_polygon[i]] != label_p[0] and D != 1:
        total_area_large-=(theta)*(sizes[largest_polygon[i]]**2)*0.5
    else:
        pass


if D != 1:
    area_fraction_large=(total_area_large)/(total_area_large+total_area_small) #area fraction of discs 
    print('area fraction: '+str(area_fraction_large))
    
else:
    pass

total_area=(area(largest_polygon, positions))

packing_fraction=(total_disc_area/total_area)
print('Packing fraction: '+str(packing_fraction))


for center, radius in zip(positions, sizes):
    x = circle(center, radius)
    plt.plot(x[:,0],x[:,1], 'b')

plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()

labels, counts = np.unique(cell_distribution, return_counts=True)
print(labels)
print(counts/np.sum(counts))

plt.bar(labels, counts/np.sum(counts), align='center')
plt.gca().set_xticks(labels)
plt.xlabel('Cell order')
plt.ylabel('Frequency')
plt.title('Cell order distribution for p='+str(P)+' for D='+str(D))
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
plt.xticks(np.arange(min(labels), max(labels)+1, 1.0))
plt.show()

mean_cell_order=round(np.sum(cell_distribution)/len(cell_distribution),5)
print('Mean cell order: '+str(mean_cell_order))

#Cell area distribution
weights = np.ones_like(cell_areas) / len(cell_areas)

counts, bins, patches = plt.hist(cell_areas, weights=weights)

plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
plt.title('Cell area distribution for p='+str(P)+' for D='+str(D))
plt.xlabel('Cell area')
plt.ylabel('Frequency')
plt.show()
mean_cell_area_main=round(np.sum(cell_areas)/len(cell_areas),5)
print('Mean cell area: '+str(mean_cell_area_main))


for order in labels:
    areas=[]

    for x in conditional_area:
        if int(x[0])==order:
            areas.append(x[1])
        else:
            pass

    print('Cell area distribution for order '+str(order)+' and p='+str(P))
    weights = np.ones_like(areas) / len(areas)
    counts, bins, patches = plt.hist(areas,weights=weights)
    print(counts)
    print(bins)

    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.xlabel('Cell area')
    plt.ylabel('Frequency')
    if order==3:
        cry=0
        if bins[0]<2:
            cry+=counts[0]
        if bins[-1]>20:
            cry+=counts[-1]
        print(cry)
        plt.text(4, 0.5, 'Cell fraction: '+str(np.round(cry,2)))
        pass
    plt.show()
    mean_cell_area=round(np.sum(areas)/len(areas),5)
    print('Mean cell area: '+str(mean_cell_area))
    
#Cell area distribution
print('Cell angle distribution for p='+str(P)+' for D='+str(D))
weights = np.ones_like(cell_angles) / len(cell_angles)

counts, bins, patches = plt.hist(cell_angles, weights=weights)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
plt.title('Cell angle distribution for p='+str(P)+' for D='+str(D))
plt.xlabel('Cell angle')
plt.ylabel('Frequency')
plt.show()
mean_cell_angle_main=round(np.sum(cell_angles)/len(cell_angles),5)
print('Mean cell angle: '+str(mean_cell_angle_main))

for order in labels:
    angles=[]

    for x in conditional_angle:
        if int(x[0])==order:
            angles.extend(x[1])
        else:
            pass
    
    weights = np.ones_like(angles) / len(angles)
    counts, bins, patches = plt.hist(angles, weights=weights)
    plt.title('Cell angle distribution for order '+str(order)+' and for p='+str(P)+' and D=3.70')
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.xlabel('Cell angle')
    plt.ylabel('Frequency')

    plt.show()
    mean_cell_angle=round(np.sum(angles)/len(angles),5)
    print('Mean cell angle: '+str(mean_cell_angle))

