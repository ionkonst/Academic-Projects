'''
Ioannis Konstantinidis
AEM: 17355
ioannisak@mail.auth.gr
'''
import copy
import random

"""
The project focuses on creating a player for the strategic board game Reversi (Othello) 
as part of the course "Advanced technics in Computer Programming." 
Reversi is a game played on an even-sized square board (e.g., 8×8). 
Players aim to capture the opponent's pieces by surrounding them horizontally, vertically, or diagonally. 
Captured pieces flip to the player's color. 
The game concludes when the board is full, or no legal moves remain, and the player with the most pieces wins.
In this script, player agent makes random moves, without a strategy
"""
class Cell:
	
	#Defining an owner attribute for the cell object. Default value is -1. Value meanings can be seen below
	def __init__(self, owner=-1):
		self.owner = owner
		
	#A method that returns the previously defined owner attribute
	def getOwner(self):
		return self.owner

	#A method that sets a new value for the owner attribute
	def setOwner(self, int):
		self.owner = int

		"""
		owner:
		-1 = empty
		0 = black
		1	= white
		2	= possible placement cell
		"""


class Reversi33:

	#Initializing the pid and size attributes. pid = player id, size = sqrt(number of squares of the playing board) 
	def __init__(self, pid=0, size=8):
		self.pid = pid
		self.size = size

	#A method that sets a new value for the pid attribute
	def setPid(self, int):
		self.pid = int

		"""
		0 black
		1	white
		"""
		
	#A method that sets the board size
	def setBoardSize(self, int):
		self.size = int
		
	#A method that returns the makers of this project, as a string
	def getPlayerName(self):
		return ('Konstantinidis')

	#The findNeighhbours method identifies the cells in which a piece can be placed and marks their owner attribute as 2. To achieve this we search, for each pid owned cell, towards the 8 different directions for empty cells connected by opponent-colored pieces.
	def findNeighbours(self, table):  #input is a list containing lists, representing board rows, containing Cell objects. Same for the methods below
		input = copy.deepcopy(table)  #we are working over a copy of the initial input, in order not to alter it in the process. Same for the methods below
		for i in range(self.size):
			for j in range(self.size):
				if input[i][j].owner == self.pid:  #identify player owned cells
					#print((i+1, j+1))  #print player owned cells

					opponent = (1 - self.pid) #if pid = 1 --> opponent = 0, if pid = 0 --> opponent = 1

					for k in range(-1, 2):  # k and l taking values -1,0,1 combining to vectors forming 8 directions
						for l in range(-1, 2):
							if (i + k) < (self.size) and (i + k) >= 0 and (j + l) < (self.size) and (j + l) >= 0: #bounds checking
								if input[i + k][j + l].owner == opponent:  #identify neighbouring opponents. If True we start searching for an empty cell at the direction of the (k,l) vector
									#print((i + k + 1, j + l + 1))  #print neighbouring opponents
	
									flag = 1  #a variable used to control the searching loop
									increment_k = k  #increment_k,increment_l: variables used as scalars of the initial direction vector. Scaling is done right after the while loop initialization
									increment_l = l
	
									while flag == 1:  #identify possible placement cells
										increment_k += k
										increment_l += l
										if (i + increment_k) > (self.size -1) or (i + increment_k) < 0 or (j + increment_l) > (self.size -1) or (j + increment_l) < 0:  #bounds checking
											break #if out of bounds we check the next direction
	
										if input[i + increment_k][j + increment_l].owner == -1:  #if cell is empty we mark it possible for placement
											input[i + increment_k][j + increment_l].owner = 2
											#print("found")
											#print((i + increment_k + 1, j + increment_l + 1))  #print possible placement cells
											flag = 0
											#print("_")
										elif input[i + increment_k][j + increment_l].owner == self.pid: #if cell is occupied by same color or already marked we pass
											flag = 0  
										elif input[i + increment_k][j + increment_l].owner == 2:
											flag = 0  
										
										
		return(input) #findNeighbours returns the initial list with the owner of the possible for piece placement cells changed to 2

	#The placeTile method picks a random cell based on the identified appropriate cells of the findNeighbours method
	def placeTile(self, table):   
		input = copy.deepcopy(table)
		
		takeover_List = []
		takeover_List = self.findNeighbours(input)   #a list with owner attributes set to 2 for the appropriate cells

		possibles_List = []
		
		for i in range(self.size):  #creating the possibles_List list containing the 'owner = 2' cells as tuples 
			for j in range(self.size):
				if takeover_List[i][j].owner == 2:
					possibles_List.append((i+1,j+1))  
		
		chosen_cell=(random.choice(possibles_List))  #picking a random cell from possibles_List
		
		if len(takeover_List) == 0:  #if no possible cells for piece placement
			chosen_cell = ()
		return(chosen_cell)  #the method returns the chosen cell as a single tuple representing its coordinates

	#The findTakeOverCells method identifies the opponent cells that must be changed to the current player's color, based on the cell he has chosen to place his piece. As with findNeighbours we will be searching in 8 different directions. The method features many variables used in a similar way with findNeighbours. However, contrary to the findNeighbours method, we care and store the intermediate steps of the loop, not a final cell 
	def findTakeOverCells(self, cell_coords, table):  #findTakeOverCells also has a 'cell_coords' tuple as an input
		input = copy.deepcopy(table) 
		
		i = cell_coords[0] - 1 #converting the tuple coords to variables appropriate for use with the table input
		j = cell_coords[1] - 1
		opponent = (1 - self.pid)

		output = []  #the output list will contain tuples representing the captured opponent cells
		
		for k in range(-1, 2):  # k and l taking values -1,0,1
			for l in range(-1, 2):
				if (i + k) < (self.size) and (i + k) >= 0 and (j + l) < (self.size) and (j + l) >= 0: #bounds checking
					#print((i+k+1,j+l+1))
					if input[i + k][j + l].owner == opponent:  #identify neighbouring opponents. If True we start searching for more opponent cells at the direction of the (k,l) vector
						#print((i + k + 1, j + l + 1))  #print neighbouring opponents
						#print(input[i + k][j + l].owner)
											
						flag = 1  
						increment_k = k
						increment_l = l
						temp_List = []
						
	
						while flag == 1:  #identify takeover cells 
							
							if (i + increment_k) > (self.size - 1) or (i + increment_k) < 0 or (j + increment_l) > (self.size - 1) or (j + increment_l) < 0: 
								break  #bounds checking
									
							if input[i + increment_k][j + increment_l].owner == opponent:  #if cell is occupied by the opponent we add the cell to a temporary list
								flag = 1       
								foo = (i+increment_k+1, j +increment_l +1)
								temp_List.append(foo)
								
							elif input[i + increment_k][j + increment_l].owner == -1:  #if we reach an empty cell we abandon this k,l direction
								flag = 0
											
							elif input[i + increment_k][j + increment_l].owner == self.pid:  #if we reach a pid owned cell we add the tuples in the temporary list to the output list
								for bar in range(len(temp_List)):
									output.append(temp_List[bar])
								flag = 0
																						
							increment_k += k  #scaling at the end of the while loop, so that we also search for opponents at the neighbouring cells
							increment_l += l
							
		return(output)  #output is a list containing tuples representing opponent pieces to be taken over

	#The applyChanges method uses all the methods above to complete a single turn for a player, given an initial board
	def applyChanges(self, table):
		input = copy.deepcopy(table)
		chosen_cell = self.placeTile(input)  #an empty cell is picked by the placeTile method
		input[chosen_cell[0]-1][chosen_cell[1]-1].owner = self.pid  #changing the owner attribute of the chosen cell to be that of pid
		
		print("chosen_cell= ", end='')  #print the chosen cell
		print(chosen_cell)
		
		takeover_List = []  #changing the owner attribute of the captured opponent cells to be that of pid
		takeover_List = self.findTakeOverCells(chosen_cell, input)
		for i in range(len(takeover_List)):
			takeover_cell = takeover_List[i]
			input[takeover_cell[0]-1][takeover_cell[1]-1].owner = self.pid
		return(input)  #applyChanges returns the board as if the player has completed his turn
		


def main():
	#following there are 4 different test_tables to test the methods above. Test by changing the table in ln. 227
	#table image3 page2
	test_table1 = [[-1, -1, -1, 1, -1, -1, -1, -1],
	               [-1, -1, -1, 1, -1, -1, -1, -1],
								 [1, 1, 1, 1, 0, -1, -1, -1],
	               [-1, -1, 1, 0, 0, 0, -1, -1],
								 [-1, -1, 1, 0, 0, -1, -1, -1],
	               [-1, -1, -1, 0, 1, 1, 1, -1],
								 [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1]]
	#table image2a page1
	test_table2 = [[-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, 0, -1, -1, -1, -1],
								 [1, 1, 1, 1, 1, -1, -1, -1],
	               [-1, -1, 1, 0, 1, -1, -1, -1],
								 [-1, -1, 1, 0, 1, -1, -1, -1],
	               [-1, -1, -1, 0, 1, 1, 1, -1],
								 [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1]]
	#initial position
	test_table3 = [[-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],=
	               [-1, -1, -1, 1, 0, -1, -1, -1],
								 [-1, -1, -1, 0, 1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1]]
	#only possible chosen cell is (8,5)
	test_table4 = [[-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
	               [-1, -1, -1, -1, -1, -1, -1, -1],
								 [0, 0, 0, 0,0, 0, 0, 0],
	               [1, 0, 1, 1, 0, 1, 0, 1],
	               [0, 0, 0, 0, 1, 1, 1, 1],
	               [0, 0, 0, 1, -1, 1, 0, 0]]

	
	board = Reversi33(0, 8)
	table = []
	for i in range(board.size):
		temp_list = []
		for j in range(board.size):
			temp_list.append(Cell(test_table4[i][j]))  #fill the owner attributes of 'test_table_x'
		table.append(temp_list)
	
	
	returned_table = board.applyChanges(table)

	#print the owner attributes of the initial table after all changes
	print('>cell coordinates \n>original owner attribute \n>owner attribute after identification of available placement cells \n>final owner attribute')
	for i in range(board.size):
		for j in range(board.size):
			print("__.__.__.__.__.__.__.__.__")
			print((i+1,j+1))
			print("original= ", end='')
			print(table[i][j].owner)
			print("original_w/neighbours= ", end='')
			print(board.findNeighbours(table)[i][j].owner)
			print("result= ", end='')
			print(returned_table[i][j].owner)

if __name__== "__main__":
    main()