from PIL import Image, ImageGrab
import pytesseract
import cv2
import numpy as np
from time import sleep,time
from pyomo.environ import ConcreteModel, Var, Objective, Constraint,ConstraintList, SolverFactory
from pyomo.environ import Binary, RangeSet, PositiveIntegers
import pyautogui #for absolutely no reason removing this breaks the program, the positioning doesn't go where it is supposed to go.
from pynput.mouse import Button, Controller

#path to the executable, because it's not on path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
#print(pytesseract.image_to_string(Image.open(img)))


def identify_numbers(img,i,j,fichier=False):
    if fichier:
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
    erode = cv2.erode(thresh, np.array((7, 7)), iterations=1)
    text = pytesseract.image_to_string(erode, config='--psm 8 -c tessedit_char_whitelist=123456789')
    #no psm mode recognises 9, it's either 0 or 2 so i have to rotate it, they find a 6 so it's a 9
    if text=="" or text=="2\n":
        if fichier:
            image = cv2.imread(img)
            image=np.rot90(image,2)
            #image2=image.transpose(Image.ROTATE_180)
            #if text=="2\n":
            #    Image.fromarray(image).show()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image=np.array(img)
            image=np.rot90(image,2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
        erode = cv2.erode(thresh, np.array((7, 7)), iterations=1)
        text2 = pytesseract.image_to_string(erode, config='--psm 8 -c tessedit_char_whitelist=123456789')
        if text2=="6\n":
            text="9"
    #cv2.imwrite("C:/Users/jeanb/OneDrive/Documents/Python/Applications mobiles avec scrpy/Sudoku/erode/{}-{}.png".format(str(i),str(j)),erode)
    #cv2.imshow("image",image)
    #cv2.imshow("gray",gray)
    #cv2.imshow("thresh",thresh)
    # cv2.imshow("erode",erode)
    # cv2.waitKey(0)
    return text

""" correct=np.array([[7,0,0,5,2,0,0,8,0],
                  [0,6,4,0,0,0,0,0,9],
                  [0,0,2,0,0,9,0,1,0],
                  [3,0,0,0,9,4,6,0,0],
                  [0,5,0,3,0,1,0,2,0],
                  [0,0,7,2,8,0,0,0,1],
                  [0,8,0,4,0,0,1,0,0],
                  [1,0,0,0,0,0,5,6,0],
                  [0,7,0,0,3,6,0,0,4]]) """

def grab_taux_pixels(x1,y1,x2,y2, couleur):
    im = ImageGrab.grab(bbox =(x1, y1, x2, y2))
    #im.show()
    px=im.load()
    total=[]
    for x in range(x2-x1):
        for y in range(y2-y1):
            if px[x,y] == couleur:
                return False
            #total.append(px[x,y])
    return True
    
#parcourir tout le sudoku
def lecture_sudoku():
    print("LECTURE")
    debut=time()
    x0,y0=1019,413
    distance=788/9
    cote=40
    lecture=[]
    for i in range(9):
        for j in range(9):
            x1, y1 = int(x0+j*distance - cote), int(y0+i*distance - cote)
            x2, y2 = int(x0+j*distance + cote), int(y0+i*distance + cote)
            im = ImageGrab.grab(bbox =(x1,y1,x2,y2))
            vide=grab_taux_pixels(int(x0+j*distance - 15), int(y0+i*distance - 20),int(x0+j*distance + 15), int(y0+i*distance + 20),(68,68,68))
            if vide:
                continue
            #im.save("C:/Users/jeanb/OneDrive/Documents/Python/Applications mobiles avec scrpy/Sudoku/image grab/{}-{}.png".format(str(i),str(j)))
            #im.show()
            #im="C:/Users/jeanb/OneDrive/Documents/Python/Applications mobiles avec scrpy/Sudoku/image grab/{}-{}.png".format(str(i),str(j))
            text=identify_numbers(im,i,j).replace("\n","")
            #text=identify_numbers(im,i,j,fichier=True).replace("\n","")
            if text=="":
                text=0
            #if int(text)!=correct[i,j]:
            #    print((i,j),text,correct[i,j])
            if text!=0:
                lecture.append((i+1,j+1,int(text)))
    print("durée de lecture : ",time()-debut)
    return lecture

def SudokuFinder(liste_nombres):
    debut=time()
    # Create concrete model
    model = ConcreteModel()

    # Set of indices
    model.I = RangeSet(1, 9)
    model.J = RangeSet(1, 9)
    model.K = RangeSet(1, 9)

    # Variables
    model.z = Var(within=PositiveIntegers)
    model.x = Var(model.I, model.J, model.K, within=Binary)

    # Arbitrary Objective Function : we simply want a feasible solution
    model.obj = Objective(expr=model.z)

    # Introducing the known numbers
    model.fixed=ConstraintList()
    for n in liste_nombres:
        model.fixed.add(expr=model.x[n[0],n[1],n[2]] == 1)

    # Every cell "(i,j)" must contain a single number "k"
    def CellUnique(model, i, j):
        return sum(model.x[i, j, k] for k in model.K) == 1

    model.cellUnique = Constraint(model.I, model.J, rule=CellUnique)


    #one number k per line
    model.line=ConstraintList()
    for i in model.I:
        for k in model.K:
            model.line.add(expr=sum(model.x[i,j,k] for j in model.J) == 1)

    #one number k per column
    model.column=ConstraintList()
    for j in model.J:
        for k in model.K:
            model.column.add(expr=sum(model.x[i,j,k] for i in model.I) == 1)


    #one number per square
    model.square=ConstraintList()
    for p in range(1,4):
        for q in range(1,4):
            for k in model.K:
                model.square.add(expr=sum(model.x[i,j,k] for i in range(3*p-2,3*p+1) for j in range(3*q-2,3*q+1)) == 1)

    # Solve the model
    sol = SolverFactory('gurobi').solve(model,tee=True)

    # CHECK SOLUTION STATUS

    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None
    if sol_json['Solver'][0]['Termination condition'] != 'optimal':
        return None

    print("Durée de résolution : ",time()-debut)
    return model.x

def PlotSudoku(x):
    # Report solution value
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    sol = np.zeros((9, 9), dtype=int)

    for i, j, k in x:
        if x[i, j, k]() > 0.5:
            sol[i - 1, j - 1] = k

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 6))
    plt.imshow(sol, interpolation='nearest', cmap=cmap)
    plt.title("Sudoku")
    plt.axis('off')

    for i, j in itertools.product(range(9),range(9)):
        plt.text(j,
                 i,
                 str(int(sol[i, j])),
                 fontsize=24,
                 ha='center',
                 va='center')

    plt.tight_layout()
    plt.show()


def input_solution(solution,liste_nombres):

    sol_numpy=np.zeros((9,9))
    for i, j, k in solution:
        if solution[i, j, k]() > 0.5:
            sol_numpy[i - 1, j - 1] = k
    print(sol_numpy)

    pos_chiffres=[1023, 1109, 1195, 1281, 1367, 1453, 1539, 1625, 1711]
    x0,y0=1019,413
    distance=788/9
    duree=0.5
    for i in range(9):
        for j in range(9):
            if (i+1,j+1,sol_numpy[i,j]) in liste_nombres:
                #print((i+1,j+1,sol_numpy[i,j]))
                continue
            mouse.position = (int(x0+j*distance),int(y0+i*distance))
            sleep(duree)
            mouse.click(Button.left)
            sleep(duree)
            mouse.position = (pos_chiffres[int(sol_numpy[i,j]) - 1], 1475)
            sleep(duree)
            mouse.click(Button.left)
            sleep(duree)
    mouse.position = (pos_chiffres[int(sol_numpy[i,j]) - 1], 1475)
    mouse.click(Button.left)


#main
compteur=0
mouse = Controller()
sleep(3)
while compteur<5:
    compteur+=1
    liste_nombres=lecture_sudoku()
    print(liste_nombres, len(liste_nombres))
    solution=SudokuFinder(liste_nombres)
    #PlotSudoku(solution)
    input_solution(solution,liste_nombres)
    sleep(4)
    mouse.position = (1365,1365)
    sleep(0.5)
    mouse.click(Button.left)
    sleep(2)
    mouse.position = (1365,1022)
    sleep(0.5)
    mouse.click(Button.left)
    sleep(0.5)
    mouse.position = (2248,735)
