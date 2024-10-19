# Sudoku app solver
 
Designed for my screen, but pixel coordinates can be changed.

How to use :
1) Connect your phone to the computer with scrpy : https://github.com/Genymobile/scrcpy
2) Open the app on your phone
3) Launch the script on your computer, it will see and click on the window of scrpy which is the same as playing on the phone.

The different steps of the script :
1) Reads the number on the screen (PIL, OpenCV, Pytesseract)
2) Solves the sudoku with linear programming (Pyomo, Gurobi)
3)  Writes the solution on the screen (Pyautogui, Pynput)
