/*
#########################################################################################
#																						#
#	    	  			    hypergraph transitive closure 								#
# nvcc -rdc=true -lineinfo -std=c++17 -Xcompiler -openmp .\progetto.cu -o progetto.exe	#
#			 			-D HIDE  : hide the output of graph								#
#			 			-D DEBUG : show information on runtime							#
#			 			-D FILE_OUT : export graph to file								#
#			 			-D MAX_THREADS : max cuda threads 								#
#			 			-D MAX_BLOCKS_A : max cuda blocks BFS							#
#			 			-D MAX_BLOCKS_AI : max cuda blocks BFS inside					#
#			 			-D MAX_BLOCKS_B : max cuda blocks succintion					#
#			 			-D TIME : enable time control	 								#
#			 			-D NO_INPUT : remove enter clic 								#
#			 			-D NTHR : number of cpu threads 								#
#			 			-D NTAB : hide the succinted graph outupt						#
#			 			-D NO_DOUBLE : to use original BFS CUDA 						#
#																						#
#########################################################################################
*/

/*
Calcolo della chiusura transitiva succinta dato un ipergrafo H, in maniera parallela ove
	se ne disponga le risorse.
	Parallelismo CPU: per ogni vertice del ipergrafo si invoca BFS; merge dei vettori 
		ottenuti dalla BFS.
	Parallelismo GPU: la BFS per trovare tutti i nipoti del nodo.
	
Calculation of the succinct transitive closure given a hypergraph H, in a parallel way where
	there is the Hardware.
	CPU parallelism: BFS is invoked for each vertex of the hypergraph; merge of vectors
		obtained from the BFS.
	GPU parallelism: the BFS to find all the grandchildren of the node.
Compile example:
	nvcc -rdc=true -D FILE_OUT -D DEBUG  -D MAX_THREADS=1024 -D MAX_BLOCKS_A=4 -D MAX_BLOCKS_A1=4 -D MAX_BLOCKS_B=16 -D TIME -D HIDE -D NO_INPUT -D NTAB -std=c++17 -lineinfo -Xcompiler -openmp -Xlinker /HEAP:0x8096 .\progetto.cu -o .\eseguibile\progetto.exe
Work:
	progetto.exe "grafo.txt"
	
Experimental:
	inside copyaa add comment around std::copy and compile without -std c++17 
	end uncomment the openmp for cycle 
	
*/


#include <cuda.h>
#include <omp.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <istream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <execution>
#include <map>
#include <utility>

//Massimo numero di threads su GPU e Block su GPU e poi su CPU

//GPU
#ifndef MAX_THREADS
#define MAX_THREADS 128
#endif
#ifndef MAX_BLOCKS_A
#define MAX_BLOCKS_A 1
#endif
#ifndef MAX_BLOCKS_AI
#define MAX_BLOCKS_AI 1
#endif
#ifndef MAX_BLOCKS_B
#define MAX_BLOCKS_B 1
#endif
//CPU
#ifdef NTHR
	int nThr = NTHR;
#else
	int nThr;
#endif

#ifdef TIME
std::chrono::high_resolution_clock::time_point begin, begin1;
std::chrono::high_resolution_clock::time_point end, end1;
unsigned long durata,durataClos,durataRead,durataSucc,durataMemory,durataMemoryT;

#endif

#ifdef DEBUG
int lineStr=1;
#endif


//gestione e cattura errori GPU
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//Struttura iperarco

typedef struct{
	int* from;					//Insieme nodi CPU
	int* from_dev;				//Insieme nodi GPU
	int len_fr;					//lunghezza dei vettori precedenti
	int to;						//Nodo di arrivo dell'iperarco
} Hyperarc;

//Struttura sorgente

typedef struct{
	int* vectors;
	int len;
	bool real;
} Hypervector;


//definizione di coppia di int

typedef std::pair<int, int> pair;



int copyaa(Hyperarc* a, Hyperarc* b, int w);
int copyaa(Hypervector* a, Hypervector* b, int w);

bool ne_compareTwoHyerarch(Hyperarc a, Hyperarc b);
bool compareTwoHyerarch(Hyperarc a, Hyperarc b);



//Hyperarch comparison function for the unique function (Equal)
bool ne_compareTwoHyerarch(Hyperarc a, Hyperarc b)
{
	if(a.to!=b.to)
	return false;
	bool ok=true;
	int i;
	for(i=0; i<min(a.len_fr,b.len_fr) && ok; i++) {
		ok=ok && a.from[i]==b.from[i];
	}
	
	if(ok) return (a.len_fr==b.len_fr);
	else return false;
}
 
//Hyperarch comparison function for the unique function (EqualLess)
bool compareTwoHyerarch(Hyperarc a, Hyperarc b)
{
	
	bool ok=true;
	int i,j=0;
	for(i=0; i<min(a.len_fr,b.len_fr) && ok; i++) {
		ok=ok && a.from[i]==b.from[i];
		j=i;
	}
	
	if(ok) return (a.len_fr<b.len_fr);
	else if(a.len_fr==b.len_fr) return a.to < b.to; else return a.from[j]<b.from[j];
}


/*   ## CPU ##
Read graph from file
*/
/*
Input:
	FILE : relative path and file name
	Vertices : pointer of pointer of integers 
	num_vertices : number of integers pointed from *Vertices
	Edges : pointer of pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from *Edges
	initial : pointer of pointer of Hypervector (set of Superset)
	num_initial : number of hypervector pointer from *initial
!!!!!!!
MODIFY:
	**Vertices, **Edges, **initial, num_edges, num_vertices, num_initial
*/
void readGraph(std::string FILE, int**& Vertices,int &num_vertices, Hyperarc**& Edges,int &num_edges, Hypervector**& initial, int &num_initial){
	std::ifstream file_graph;
	std::string line, pref, from;
	int idxE = 0, idxV = 0, len_fr, to, temp, temp1,temp2, idxEe=0;
	std::map<std::string, pair> temporaneo;
	bool succinto = false;
	int num_real_source;
	
	file_graph.open(FILE, std::ios::in);
	if (file_graph.is_open())
	{
		while ( std::getline(file_graph,line) )
		{
			pref = line.substr(0,3);
			if(pref=="INI"){
				num_vertices = std::stoi(line.substr(3, (int)line.find(",")-3));
				num_edges = std::stoi(line.substr(line.find(",")+1));
				
				printf("Vertici %d, Edges %d ", num_vertices, num_edges);
				
				*Vertices = (int*) malloc(sizeof(int)*num_vertices);
				*Edges = (Hyperarc*) malloc(sizeof(Hyperarc)*num_edges);
				
				printf("OK\n");
				#ifdef DEBUG
				lineStr++;
				#endif
			}else if(pref == "SUC"){
				temp  = line.find(",")+1;
				temp1 = line.find(",",temp+1);
				temp2 = line.find(",",temp1+1);
								
				num_vertices 	= std::stoi(line.substr(3, (int)line.find(",")-3));
				num_edges    	= std::stoi(line.substr(temp1+1, temp2-temp1));
				num_initial  	= std::stoi(line.substr(temp2+1));
				num_real_source = std::stoi(line.substr(temp, temp1-temp));
				printf("Vertici %d, Edges %d, Sources %d (%d)", num_vertices, num_edges, num_real_source, num_initial);
				
				*Vertices = (int*) malloc(sizeof(int)*num_vertices);
				*initial = (Hypervector*) malloc(sizeof(Hypervector)*num_real_source);
				*Edges = (Hyperarc*) malloc(sizeof(Hyperarc)*num_edges);
				
				
				printf("OK\n");
				#ifdef DEBUG
				lineStr++;
				#endif
			}else if(pref == "MAT"){
				succinto = true;	
			}else if(pref == "(SE"){
				temp = line.find("}");
				temp2 = line.find("{");
				from = (line.substr(temp2+1, temp-(temp2+1)));
				temp1 = line.find(",",temp+1);
				len_fr = std::stoi(line.substr(temp1+1, temp));
				
				(*initial)[idxE] = {(int*)malloc(sizeof(int)*len_fr), len_fr, true};
				temp = 0;
				
				
				auto pa = std::make_pair(len_fr,idxE);
				temporaneo.insert(make_pair(from,pa));
				
			
				for(int i=0; i<len_fr; i++){
					temp1 = from.find(",", temp);
					if(temp==-1)
						(*initial)[idxE].vectors[i] = std::stoi(from.substr(temp));
					else
						(*initial)[idxE].vectors[i] = std::stoi(from.substr(temp,temp1));
					temp=temp1+1;
					
				}
				
				for(int i=0; i<len_fr; i++)
					printf("%d ",(*initial)[idxE].vectors[i]);
				printf(", %d\n",(*initial)[idxE].real?1:0);
				
				idxE++;
				
			}else if(pref == "RGS" && succinto){
				temp  = 4;
				temp1 = line.find(",",temp);
				temp2 = 0;
				while(line.substr(temp)!="REG"){
					printf("%s ",line.substr(temp).c_str());
					if(line.substr(temp,1)=="1"){
						(*Edges)[idxV] = {(*initial)[idxEe].vectors,NULL, (*initial)[idxEe].len, temp2};
						idxV++;
					}
					temp2++;
					
					temp = line.find(",",temp+1)+1;
					printf("%d\n",temp);
					if(temp == 0) temp = line.find(" ",temp+4)+1;
				}
				printf(" %d\n",idxV);
				idxEe++;
			}else if(pref == "(HA"){
				temp = line.find("}");
				temp2 = line.find("{");
				from = (line.substr(temp2+1, temp-(temp2+1)));
				temp1 = line.find(",",temp+1);
				temp2 = line.find(",", temp1+1);
				len_fr = std::stoi(line.substr(temp1+1, temp2-temp));
				
				to = std::stoi(line.substr(temp2+1, line.find(")")-(temp2+1)));
				(*Edges)[idxE] = {(int*)malloc(sizeof(int)*len_fr), NULL, len_fr, to};
				temp = 0;
				if(temporaneo.find(from)==temporaneo.end()){
					auto pa = std::make_pair(len_fr,idxE);
					temporaneo.insert(make_pair(from,pa));
					
				
					for(int i=0; i<len_fr; i++){
						temp1 = from.find(",", temp);
						if(temp==-1)
							(*Edges)[idxE].from[i] = std::stoi(from.substr(temp));
						else
							(*Edges)[idxE].from[i] = std::stoi(from.substr(temp,temp1));
						temp=temp1+1;
						
					}
				}else{
					auto it = temporaneo.find(from)->second;
					(*Edges)[idxE].from = (*Edges)[(it.second)].from;
					
				}
				idxE++;
			}else if(pref == "(VE"){
				(*Vertices)[idxV] = std::stoi(line.substr(3, line.find(")")-1));
				idxV ++;
			}
		}
		
		if(num_real_source != temporaneo.size()){
			num_initial = temporaneo.size();
			*initial = (Hypervector*) malloc(sizeof(Hypervector)*num_initial);
			
		}else{
			num_initial = num_real_source;
		}
		int itera=0,ttop;
		pair tt;
		
		
		bool * manca = (bool*) malloc(sizeof(bool)*num_vertices);
		for(int i=0; i<num_vertices; i++) manca[i]=true;
		
		if(!succinto){
			for(auto it=temporaneo.begin(); it!=temporaneo.end(); it++){
				tt = it->second;
					(*initial)[itera] = {(*Edges)[tt.second].from,tt.first, true};
					for(int i=0; i<(*Edges)[tt.second].len_fr; i++){
						manca[(*Edges)[tt.second].from[i]] = false;
					}
					itera++;
				
			}
		}else{
			for(int i=0; i<num_real_source; i++){
				for(int j=0; j<(*initial)[i].len; j++){
					printf("%d ",(*initial)[i].vectors[j]);
					manca[(*initial)[i].vectors[j]]=false;
					
				}
			}
			for(int i=0; i<num_vertices; i++) (*Vertices)[i] = i;
				
		}
		itera = 0;
		for(int i=0; i<num_vertices; i++) if(manca[i]) itera++;
		
		Hypervector* tempo = (Hypervector*) malloc(sizeof(Hypervector)*(itera+num_initial));
		
		copyaa(tempo,*initial,num_initial);
		
		ttop=0;
		for(int i=0; i<num_vertices; i++){
			if(manca[i]){
				tempo[num_initial+ttop] =  {new int{i},1,false};
				ttop++;
			}
		}
		
	
		
		num_initial += itera;
		free(*initial);
		
		
		*initial = (Hypervector*) malloc(sizeof(Hypervector)*num_initial);
		copyaa(*initial, tempo, num_initial);
		
			
		file_graph.close();
		printf("Superset: %d ok\n",num_initial);
		#ifdef DEBUG
		lineStr++;
		#endif
		
	}
	
}

/*   ## CPU ##
Write graph from file, succinted
*/
/*
Input:
	matrixSuc : matrix of Graphs
	num_vertices : number of integers pointed from Vertices
	Sources : array of Hypervector (Source set)
	num_sources : number of Hypervector pointed from Sources
	num_edges : number of edges of graph
	FILE : relative path and file name
*/
void writeGraph(bool** matrixSuc, int num_vertices, Hypervector* Sources, int num_source, int num_edges, std::string FILE){
	std::ofstream myFile;
	int num_source_real = 0;
	
	for(int i=0; i<num_source; i++) if(Sources[i].real) num_source_real++;
	
	myFile.open(FILE);
	myFile << "SUC " << num_vertices <<"," << num_source_real << "," << num_edges << "," << num_source << "\n";
	#ifdef DEBUG
	lineStr++;
	#endif
	for(int i=0; i<num_source; i++){
		if(Sources[i].real){
			myFile << "(SE {";
			for(int j=0; j<Sources[i].len; j++){
				myFile << Sources[i].vectors[j];
				if(j!=Sources[i].len-1)
					myFile << ",";
				
			}
			myFile <<"},"<< Sources[i].len <<")\n";
		}
	}
	myFile <<"MAT\n";
	for(int i=0; i<(num_source); i++){
		if(Sources[i].real){
			myFile << "RGS ";
			for(int j=0; j<num_vertices; j++){
				myFile << matrixSuc[i][j]? 1:0;
				if(j!=num_vertices-1) myFile << ",";
			}
			myFile <<" REG\n";
		}
	}
	
	
	
	myFile.close();
}	


#ifdef TIME
/*   ## CPU ##
Write operation time to file
*/
/*
Input:
	num_vertices : number of integers pointed from Vertices
	num_edges : number of hyperarcs pointed from Edges
*/
void writeTime(int num_vertices, int num_edges, int num_source, int num_edge_final, int num_source_total){
	std::ofstream myFile;
	myFile.open("TimeSave.txt", std::fstream::app);
	myFile << "<G ["<<MAX_BLOCKS_A<<"] ["<<MAX_BLOCKS_AI<<"] " << num_vertices <<"," << num_edges <<","<< num_source <<","<< num_edge_final<<","<< num_source_total<<" ("<<durata<<" ms),("<<durataRead<<" ms),("<<durataClos<<" ms),("<<durataMemory<<" ms),("<<durataSucc<<" ms)>\n";
	myFile.close();
}
#endif
/*	## GPU ##
Confront hyperarcs superset with node's array  
*/
/*
Input:
	hyperarcs : array of node inside hyperarcs
	vectors : array of nodes
	minLen : min(len(hyperarcs),len(vectors))
Output:
	if arrays is equal
*/
__device__ bool equalHyperVector(int * hyperarcs, int * vectors, int minLen){
	bool ok=true;
	for(int i=0; i<minLen && ok; i++){
		ok=hyperarcs[i]==vectors[i];
	}
	return ok;
}

/*	## GPU ##
Confront hyperarcs superset with node's array  
*/
/*
Input:
	hyperarcs : array of node inside hyperarcs
	vectors : array of nodes
	len1 : len(hyperarcs)
	len2 : len(vectors)
Output:
	if arrays is equal
*/
__device__ bool equalHyperVectorL(int * hyperarcs, int * vectors, int len1, int len2){
	bool ok=true;
	if(len1!=len2) return false;
	for(int i=0; i<len1 && ok; i++){
		ok=hyperarcs[i]==vectors[i];
	}
	return ok;
}


/*   ## GPU ##
Update the ausiliary structur of BFS
*/
/*
Input:
	len : number of integers pointed from every array
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
	next : check wheater to proceed or not
	end : to share the end of operations
!!!!!!
MODIFY:
	Frontier, Visited, next, FrontierUpdate, end
*/
__global__ void bfs_update(int len, bool * Frontier, bool * FrontierUpdate, bool * Visited, int * next){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int thidI = 0;
		
	__syncthreads();
	
	for(int Pass=0; Pass<ceilf((len/(blockDim.x)))+1; Pass++){
		thidI = thid + Pass*(gridDim.x*blockDim.x );
		if(thidI<len){
			
			if(FrontierUpdate[thidI]){
				Frontier[thidI] = true;
				Visited[thidI] = true;
				*next = 1;
				FrontierUpdate[thidI] = false;
				
			}
		}
	}
	__syncthreads();
	
}



/*   ## CPU ##
Force the copy of array inside another (struct of Hyperarcs)
*/
/*
Input:
	a : pointer of Hyperarcs (to)
	b : pointer of Hyperarcs (from)
	w : size of a and b 
return:
	if the copy ended 0, anything else
!!!!!!
MODIFY:
	a
*/
int copyaa(Hyperarc* a, Hyperarc* b, int w){
	
	//std::memcpy(a,b,w*sizeof(Hyperarc));
	
	//using std c++17
	std::copy(std::execution::par,b,b+w,a);
	
	std::sort(a, a + w, compareTwoHyerarch);
	auto discard=std::unique( a, a + w , ne_compareTwoHyerarch);
	return 0;
}

/*   ## CPU ##
Force the copy of array inside another (struct of Hypervector)
*/
/*
Input:
	a : pointer of Hypervector (to)
	b : pointer of Hypervector (from)
	w : size of a and b 
return:
	if the copy ended 0, anything else
!!!!!!
MODIFY:
	a
*/
int copyaa(Hypervector* a, Hypervector* b, int w){
	
	//std::memcpy(a,b,w*sizeof(Hypervector));
	
	//using std c++17
	std::copy(std::execution::par,b,b+w,a);
	
	return 0;
}

/*   ## GPU ##
Device includes element in a array
*/
/*
Input:
	x : element to find
	arr : array where find x
	len : size of arr
return:
	true if element exist
!!!!!!
*/
__device__ bool includes(int x, int * arr, int len){
	bool ok=false;
	for(int i=0; i<len && !ok; i++){
		ok = arr[i]==x;
	}
	return ok;
}

/*   ## GPU ##
Find neighbors during one BFS step
*/
/*
Input:
	adjMatrix : hyper-graph adjacency matrix
	idVert : last vertices visited
	num_vertices : number of vertices in the graph
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
	Sources : list of all sources
	num_source : size of Sources array
!!!!!!
MODIFY:
	Cost, FrontierUpdate
*/
__global__  void neighOp_M(bool ** adjMatrix, int idVert, int num_vertices,bool * FrontierUpdate, bool * Visited, int * Cost, Hypervector * Sources, int num_source){
	int thid = (blockIdx.x*blockDim.x)+threadIdx.x;
	int thidI;
	int so, ve;
			
	for(int Pass=0; Pass<ceilf((num_source*num_vertices/(gridDim.x*blockDim.x)))+1; Pass++){
		thidI = thid + Pass*(gridDim.x*blockDim.x);
		
		if(thidI<num_source*num_vertices){
			so = thidI/num_vertices;
			ve = thidI%num_vertices;
			if(adjMatrix[so][ve] && includes(idVert, Sources[so].vectors, Sources[so].len)) 
				if(Visited[ve]==false || Cost[ve]==0){
			
						Cost[ve] = Cost[idVert]+1;
						FrontierUpdate[ve] = true;
					}		
		
		}
	}
	__syncthreads();
	
}


/*   ## GPU ##
BFS step
*/
/*
Input:
	adjMatrix : hyper-graph adjacency matrix
	num_vertices : number of vertices in the graph
	num_source : size of Sources array
	Frontier : the Frontier of BFS
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
	Sources : list of all sources
	end : to share the end of operations
!!!!!!
MODIFY:
	Cost, FrontierUpdate, Frontier, Visited, end
*/

__global__ void bfs_M(bool ** adjMatrix, int num_vertices, int num_source, bool * Frontier, bool * FrontierUpdate, bool * Visited, int * Cost, Hypervector * Sources){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int thidI;
	
	#ifdef NO_DOUBLE
	int so,ve;
	#endif
		
	__syncthreads();
	
	for(int Pass=0; Pass<ceilf(num_vertices/(gridDim.x*blockDim.x))+1; Pass++){
		thidI = thid + Pass*(gridDim.x*blockDim.x);
	
		if(thidI < num_vertices)
			if(Frontier[thidI]==true){
				Frontier[thidI]=false;
				Visited[thidI]=true;
				
				#ifndef NO_DOUBLE
				neighOp_M<<<MAX_BLOCKS_AI, MAX_THREADS>>>(adjMatrix,thidI, num_vertices, FrontierUpdate, Visited, Cost, Sources, num_source);
				#else
					for(int NN=0; NN<((num_source*num_vertices)); NN++){
		
						so = NN/num_vertices;
						ve = NN%num_vertices;
						if(adjMatrix[so][ve] && includes(thidI, Sources[so].vectors, Sources[so].len)) 
							if(Visited[ve]==false || Cost[ve]==0){
						
									Cost[ve] = Cost[thidI]+1;
									FrontierUpdate[ve] = true;
								}		
					
					}
				#endif
			}
	}
	__syncthreads();
}



/*   ## CPU ##
Succint closure with BFS visit from sourceStart (Source)
*/
/*
Input:
	adjMatrix_DEV : hyper-graph adjacency matrix (Device pointer)
	Set: list of all sources
	Set_DEV: list of all sources (Device pointer)
	num_vertices : number of vertices in the graph
	num_source : size of Sources array
	sourceStart : the id of source, where the BFS begins
Output:
	list of node visited

*/
bool * gpu_bfs_suc(bool ** adjMatrix_DEV, Hypervector *Set, Hypervector *Set_DEV, int num_vertices, int num_source, int sourceStart, bool TOK){
	bool * ret;
				
	int * Cost_HOS, *Cost_DEV;
	bool * Frontier_HOS, *Frontier_DEV;
	bool * FrontierUpdate_HOS, *FrontierUpdate_DEV;
	bool * Visited_HOS, *Visited_DEV;
	
	int next_HOS, *next_DEV;
	
	
	#ifdef TIME
	if(TOK==0)
		begin1 = std::chrono::high_resolution_clock::now();
	#endif
	Cost_HOS = (int*) malloc(sizeof(int)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Cost_DEV, sizeof(int)*num_vertices));
	
	Frontier_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Frontier_DEV, sizeof(bool)*num_vertices));
	
	FrontierUpdate_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&FrontierUpdate_DEV, sizeof(bool)*num_vertices));
	
	Visited_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Visited_DEV, sizeof(bool)*num_vertices));
	
	gpuErrchk(cudaMalloc((void**) &next_DEV, sizeof(int)));	
	
		
	ret = (bool*) malloc(sizeof(bool) * num_vertices);
	
		
	gpuErrchk(cudaGetLastError());
	
	
	
	for(int i=0; i<num_vertices; i++){
		ret[i] = false;
		Cost_HOS[i] = -1;
	}
	
	for(int i=0; i<num_vertices; i++){
		Frontier_HOS[i] = false;
		FrontierUpdate_HOS[i] = false;
		Visited_HOS[i] = false;
	}
	
	
	
	for(int i=0; i<Set[sourceStart].len; i++){
		Cost_HOS[Set[sourceStart].vectors[i]] 		= 0;
		Frontier_HOS[Set[sourceStart].vectors[i]] 	= true;
		Visited_HOS[Set[sourceStart].vectors[i]] 		= true;
		
					
	}
	
	
	cudaMemcpy(Cost_DEV, Cost_HOS, sizeof(int)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(Frontier_DEV, Frontier_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(FrontierUpdate_DEV, FrontierUpdate_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(Visited_DEV, Visited_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	gpuErrchk(cudaGetLastError());
	
	#ifdef TIME
	if(TOK==0){
		end1 = std::chrono::high_resolution_clock::now();
		durataMemoryT = std::chrono::duration_cast<std::chrono::milliseconds>( end1 - begin1).count();
	}
	#endif
	
	
	
	next_HOS = 1;
	while(next_HOS==1){
		next_HOS = 0;
		gpuErrchk(cudaMemcpy(next_DEV, &next_HOS, sizeof(int), cudaMemcpyHostToDevice));
		
		
		bfs_M<<<MAX_BLOCKS_A, min(num_vertices, MAX_THREADS) >>>(adjMatrix_DEV, num_vertices, num_source, Frontier_DEV, FrontierUpdate_DEV, Visited_DEV, Cost_DEV, Set_DEV);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
	
				
		bfs_update<<<MAX_BLOCKS_A, min(num_vertices, MAX_THREADS) >>>(num_vertices, Frontier_DEV, FrontierUpdate_DEV, Visited_DEV, next_DEV);
		cudaDeviceSynchronize();
		gpuErrchk(cudaGetLastError());
	
		gpuErrchk(cudaMemcpy(&next_HOS, next_DEV , sizeof(int), cudaMemcpyDeviceToHost));
		
	}

	
	cudaMemcpy(Cost_HOS, Cost_DEV, sizeof(int)*num_vertices, cudaMemcpyDeviceToHost);
	
	for(int i=0; i<num_vertices; i++)
		if(Cost_HOS[i]>0){
			ret[i]=true;
		}
	
	
	
	gpuErrchk(cudaFree(next_DEV));
	gpuErrchk(cudaFree(Frontier_DEV));
	gpuErrchk(cudaFree(FrontierUpdate_DEV));
	gpuErrchk(cudaFree(Visited_DEV));
	gpuErrchk(cudaFree(Cost_DEV));
	
	gpuErrchk(cudaGetLastError());
	
	return ret;
}



/*   ## CPU ##
Prepare the data and lunch BFS
*/
/*
Input:
	MatrixAdj : hyper-graph adjacency matrix 
	Set: list of all sources
	num_source : number of sources in hyper-graph
	num_vertices : number of integers pointed from Vertices
	num_edges : number of hyperarcs in hyper-graph
Output:
	Adjacency Matrix of transitive succint closure of MatrixAdj
!!!!!!
MODIFY:
	Edges
!!!!!!
MULTI PARALLELISM OPENMP
!!!!!!
Allocate temporary copy of MatrixAdj, one in RAM and other in VRAM
*/
bool ** gpu_trans_succ(bool ** MatrixAdj, Hypervector * Set, int num_source, int num_vertices, int &num_edges){
	bool ** adjMatrix;
	bool ** adjMatrix_DEV;
	
	int * from;
	bool * fromb;
	
	Hypervector * set_DEV, * set_HOS;
	
	int len_frT, sum=0;
	int totOp=0, totSo=0;
	
	std::string completo = "";
	
	#ifdef TIME
	begin1 = std::chrono::high_resolution_clock::now();
	#endif
	
	adjMatrix = (bool**) malloc(sizeof(bool*) * num_source);
	gpuErrchk(cudaMalloc((void**) &adjMatrix_DEV, sizeof(bool*) * num_source));
	
	set_HOS = (Hypervector*) malloc(sizeof(Hypervector)*num_source);
	gpuErrchk(cudaMalloc((void**)&set_DEV, sizeof(Hypervector)*num_source));
	gpuErrchk(cudaGetLastError());
		
	fromb = (bool*) malloc(sizeof(bool)*num_vertices);
	
	for(int i=0; i<num_source; i++){
		gpuErrchk(cudaMalloc(&adjMatrix[i], sizeof(bool)*num_vertices));
		gpuErrchk(cudaMemcpy(adjMatrix[i], MatrixAdj[i], sizeof(bool)*num_vertices, cudaMemcpyHostToDevice));
	}
	
	gpuErrchk(cudaMemcpy(adjMatrix_DEV, adjMatrix, sizeof(bool*)*num_source, cudaMemcpyHostToDevice));
	
	
	
	gpuErrchk(cudaGetLastError());
	
	for(int i=0; i<num_source; i++){	
		len_frT = (Set)[i].len;
		gpuErrchk(cudaMalloc((void**)&from, sizeof(int) * len_frT));
		gpuErrchk(cudaMemcpy(from, (Set)[i].vectors, sizeof(int)* len_frT, cudaMemcpyHostToDevice));
		(set_HOS)[i].vectors = from;
		set_HOS[i].len = Set[i].len;
	}	
	gpuErrchk(cudaMemcpy(set_DEV, set_HOS, sizeof(Hypervector)*num_source, cudaMemcpyHostToDevice));
	free(set_HOS);
	
	gpuErrchk(cudaGetLastError());
	
	from = (int*) malloc(sizeof(int)*nThr);
	
	
	#ifdef TIME
	end1 = std::chrono::high_resolution_clock::now();
	durataMemory = std::chrono::duration_cast<std::chrono::milliseconds>( end1 - begin1).count();
	#endif
	
	printf("Start BFS visit...\n");
	#ifdef DEBUG
	lineStr+=1;
	#endif
	
	
	#pragma omp parallel shared(adjMatrix_DEV, adjMatrix, set_DEV,Set, from, num_vertices, num_source, completo, totOp, totSo) private(fromb) num_threads(nThr)
	{
		int work_x_thr  = ceil(num_source/(nThr));
		int sx 		    = omp_get_thread_num() * work_x_thr;
		int ex 		    = sx + work_x_thr;
		int t_num_edges = 0;
		int num_att=0;
				
		from[omp_get_thread_num()] = 0;
		
		#ifdef DEBUG
		printf("thread %d: %d-%d (%d)\n",omp_get_thread_num(),sx,ex,work_x_thr);
		#pragma omp atomic
		lineStr+=1;
		#endif
		
		if(omp_get_thread_num()==nThr-1) ex = num_source;
		
		#pragma omp barrier
		for(int i=sx; i<ex; i++){
			if((Set[i].real)){
				num_att = i;
				fromb = gpu_bfs_suc(adjMatrix_DEV, Set, set_DEV, num_vertices, num_source, num_att, omp_get_thread_num()==0); 
				
				t_num_edges = 0;
				
				adjMatrix[num_att] = (bool*)malloc(sizeof(bool)*num_vertices);
				for(int j=0; j<num_vertices; j++){
					if(fromb[j]) t_num_edges++;
					adjMatrix[num_att][j] = fromb[j];
				}
				
				from[omp_get_thread_num()] += t_num_edges;
				
				free(fromb);
				#ifdef TIME
				if(omp_get_thread_num()==0)
					durataMemory+=durataMemoryT;
				 #endif
			}
			#ifdef DEBUG
			#pragma omp atomic
			totSo+=1;
			printf("\033[%d;0HCompletato %d/%d  (%s ) [%d/%d]           ",lineStr,totOp,nThr,completo.c_str(),totSo,num_source);
			#endif
			
		}
		
		#ifdef DEBUG
		completo+=" "+std::to_string(omp_get_thread_num());
		#pragma omp atomic
		totOp+=1;
		printf("\033[%d;1HCompletato %d/%d  (%s ) [%d/%d]           ",lineStr,totOp,nThr,completo.c_str(),totSo,num_source);
		#endif
	}
	
	
	
	#pragma omp barrier
	
	#ifdef DEBUG
	printf("\nEnd BFS visit\n");
	lineStr+=3;
	#endif
	num_edges = 0;
	
	#pragma omp parallel for reduction (+:sum)
	for (int i=0;i<nThr;i++)
	  sum=sum+from[i];
  
	for(int i=0; i<num_source; i++){
		if(Set[i].real){
			for(int j=0; j<num_vertices; j++){
				adjMatrix[i][j] = adjMatrix[i][j] || MatrixAdj[i][j];
			}
		}
	}
	
	num_edges = sum;
	
	
	gpuErrchk(cudaFree(adjMatrix_DEV));
	gpuErrchk(cudaFree(set_DEV));
	
	
	
	return adjMatrix;
}

/*   ## GPU ##
Create a adjacent's matrix with parallelism
*/
/*
Input:
	Vertices : list of vertices of hyper-graph
	Edges : list of Hyperarc of hyper-graph
	Source : list of Source of hyper-graph
	num_vertices : number of vertices
	num_edges : number of hyperarcs
	num_source : number of sources
	retMatrix : adjMatrix calculate
!!!!!!!
MODIFY
	retMatrix
*/
__global__ void succintaKernel(int *Vertices, Hyperarc * Edges, Hypervector * Source, int num_vertices, int num_edges, int num_source, bool ** retMatrix){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int thidI;
	
	
	__syncthreads();
	
	
	
	for(int Pass=0; Pass<ceilf((num_edges)/(gridDim.x*blockDim.x))+1; Pass++){
		thidI = thid + Pass*(gridDim.x*blockDim.x);
		if(thidI<(num_edges)){
			for(int i=0; i<num_source; i++){
									
				if(i<num_source){
					if(equalHyperVectorL(Edges[thidI].from_dev, Source[i].vectors, Edges[thidI].len_fr, Source[i].len)) {
						retMatrix[i][Edges[thidI].to] = true;
					
					}
					
				
				}
			}
		}
	}
	
	__syncthreads();
	
	
}


/*   ## CPU ##
Prepare and launch the kernel to create adjacency matrix from hyper-graph
*/
/*
Input:
	Vertices : list of vertices of hyper-graph
	Edges : list of Hyperarc of hyper-graph
	Set : list of Source of hyper-graph
	num_vertices : number of vertices
	num_edges : number of hyperarcs
	num_source : number of sources
Output:
	Adjacency Matrix of hyper-graph
!!!!!!!
Allocate temporary copy of MatrixAdj, one in RAM and other in VRAM
*/
bool ** succinta(int *Vertices, Hyperarc *Edges, Hypervector *Set, int num_vertices, int num_edges, int num_source){
	bool ** adjMatrix = (bool**) malloc(sizeof(bool*) * num_source);
	bool ** adjMatrix_DEV;
	
	Hypervector * set_DEV, * set_HOS;
	
	Hyperarc * Edges_DEV;
	int * Vertices_DEV;
	int * from;
	bool * fromb;
	
	printf("Succint start\n");
	#ifdef DEBUG
	lineStr+=1;
	#endif
	
	gpuErrchk(cudaMalloc((void**) &adjMatrix_DEV, sizeof(bool*) * num_source));

	gpuErrchk(cudaGetLastError());
	for(int i=0; i<num_source; i++){
		gpuErrchk(cudaMalloc(&adjMatrix[i], sizeof(bool)*num_vertices));
	}
	
	gpuErrchk(cudaGetLastError());
	
	cudaMemcpy(adjMatrix_DEV, adjMatrix, sizeof(bool*)*num_source, cudaMemcpyHostToDevice);
	
	gpuErrchk(cudaGetLastError());
	
	
	gpuErrchk(cudaMalloc((void**)&Vertices_DEV, sizeof(int)*num_vertices));
	gpuErrchk(cudaMalloc((void**)&Edges_DEV, sizeof(Hyperarc)*num_edges));
	gpuErrchk(cudaMalloc((void**)&set_DEV, sizeof(Hypervector)*num_source));
	
	set_HOS = (Hypervector*) malloc(sizeof(Hypervector)*num_source);
	
	cudaMemcpy(Vertices_DEV, Vertices, sizeof(int)*num_vertices, cudaMemcpyHostToDevice);
	
	
	gpuErrchk(cudaGetLastError());
	
	int len_frT = 0;
	for(int i=0; i<num_edges; i++){	
		len_frT = (Edges)[i].len_fr;
		gpuErrchk(cudaMalloc((void**)&from, sizeof(int) * len_frT));
		
		gpuErrchk(cudaMemcpy(from, (Edges)[i].from, sizeof(int)* len_frT, cudaMemcpyHostToDevice));
		
		
		(Edges)[i].from_dev = from;
		
	}	
		
	gpuErrchk(cudaMemcpy(Edges_DEV, Edges, sizeof(Hyperarc)*num_edges, cudaMemcpyHostToDevice));
	
	
	len_frT = 0;
	for(int i=0; i<num_source; i++){	
		len_frT = (Set)[i].len;
		gpuErrchk(cudaMalloc((void**)&from, sizeof(int) * len_frT));
		
		gpuErrchk(cudaMemcpy(from, (Set)[i].vectors, sizeof(int)* len_frT, cudaMemcpyHostToDevice));
		
		
		(set_HOS)[i].vectors = from;
		set_HOS[i].len = Set[i].len;
	}	
	gpuErrchk(cudaMemcpy(set_DEV, set_HOS, sizeof(Hypervector)*num_source, cudaMemcpyHostToDevice));
	
	free(set_HOS);
	
	gpuErrchk(cudaGetLastError());
		
	printf("Prepare all\n");
	#ifdef DEBUG
	lineStr+=1;
	#endif
		
	succintaKernel<<<MAX_BLOCKS_B, MAX_THREADS>>>(Vertices_DEV, Edges_DEV, set_DEV, num_vertices, num_edges, num_source, adjMatrix_DEV); 
	cudaDeviceSynchronize();
		
	gpuErrchk(cudaGetLastError());
	
	
	printf("Succinted\n");
	#ifdef DEBUG
	lineStr+=1;
	#endif
	cudaMemcpy(adjMatrix, adjMatrix_DEV, sizeof(bool)* num_source, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaGetLastError());
	
		
	for(int i=0; i<num_source; i++){
		fromb = (bool*) malloc(sizeof(bool) * num_vertices);
		cudaMemcpy(fromb,adjMatrix[i], sizeof(bool)* num_vertices, cudaMemcpyDeviceToHost);
	
		adjMatrix[i] = (bool*) malloc(sizeof(bool)*num_vertices);
		std::copy(std::execution::par,fromb,fromb+num_vertices,adjMatrix[i]);	
	}
	
	
		
	cudaFree(adjMatrix_DEV);
	
	gpuErrchk(cudaFree(Edges_DEV));
	gpuErrchk(cudaFree(Vertices_DEV));
	gpuErrchk(cudaFree(set_DEV));
	gpuErrchk(cudaGetLastError());
		
	return adjMatrix;
}


/*   ## CPU ##
main function
read graph and initialize it, lunch gpu_bfs, write graphs
(writing on console or file depends on the type of compilation)
*/
/*
Input:
	args = { name_program, [name_graphs, -nT=<numberOfOmpThreads>] }
*/
int main(int argn, char ** args){
	
	system("cls");
	
	Hyperarc** Edges	  = (Hyperarc**) malloc(sizeof(Hyperarc*));
	Hypervector** initial = (Hypervector**) malloc(sizeof(Hypervector*));
	int ** Vertices		  = (int**) malloc(sizeof(int*));
	
	bool ** MatrixAdj;
	
	int num_vertices=0, num_edges=0, num_initial=0;
	
	#ifdef FILE_OUT
	#ifdef TIME
	int  num_edges_init=0, num_real_initial=0;
	#endif 
	#endif
	bool argg=false,thName=false;
	std::string file_name="                    ";
	
	if(argn>1){
		for(int i=0; i<argn; i++)
			if(std::string(args[i]).substr(0,4) == "-nT="){
				nThr = std::stoi(std::string(args[i]).substr(4));
				argg=true;
			}
			else
				if(std::string(args[i]).substr(0,3) == "--h"){
					printf("Succinct transitive closure of hypergraphs\n  progetto.exe <option> <name of graphs>\n\nOption:\n\t-nT <number of CPU Threads>");
				}else if(std::string(args[i]).substr(0,2) == "-o"){
					file_name = args[i+1];
					thName = true;
				}else if(!thName)
					file_name = args[i];
		if(!argg){
			#ifndef NTHR
				#pragma omp parallel shared(nThr)
				{
					if(omp_get_thread_num()==0)
						nThr = omp_get_max_threads();
				}
			#endif
		}
		#ifdef TIME
		begin = std::chrono::high_resolution_clock::now();
		#endif
		readGraph(file_name,Vertices,num_vertices, Edges, num_edges, initial, num_initial);
		
		#ifdef TIME
		end = std::chrono::high_resolution_clock::now();
		#endif
			
	}else{
		
		#ifdef TIME
		begin = std::chrono::high_resolution_clock::now();
		#endif
		readGraph("prova.txt",Vertices, num_vertices, Edges, num_edges, initial, num_initial);
		
		#ifdef TIME
		end = std::chrono::high_resolution_clock::now();
		#endif
	}
	
	#ifdef FILE_OUT
	#ifdef TIME
	num_edges_init = num_edges;
	#endif
	#endif
	
	#ifdef TIME
	durataRead = std::chrono::duration_cast<std::chrono::milliseconds>( end - begin ).count();
	#ifdef DEBUG
	printf("reading time: %lu ms\n",durataRead);
	lineStr++;
	printf("CPU threads: %d\nGPU Block:%d [%d] - %d\nGPU threads: %d\n", nThr,MAX_BLOCKS_A,MAX_BLOCKS_AI,MAX_BLOCKS_B,MAX_THREADS);
	lineStr+=3;
	#endif
	#endif
	
	#ifndef HIDE
	printf("Hyperarc: %d\n",num_edges);
	#ifdef DEBUG
	lineStr++;
	#endif
	for(int i=0; i<(num_edges); i++){
		printf("(HA {");
		for(int j=0; j<(*Edges)[i].len_fr; j++){
			printf("%d",(*Edges)[i].from[j]);
			if(j!=((*Edges)[i].len_fr)-1) printf(",");
		}
		printf("} %d)\n",(*Edges)[i].to);
		#ifdef DEBUG
		lineStr++;
		#endif
	}
	
	printf("Vertices: %d\n",num_vertices);
	#ifdef DEBUG
	lineStr++;
	#endif
	for(int i=0; i<num_vertices; i++){
		printf("(VE %d)\n",(*Vertices)[i]);
		#ifdef DEBUG
		lineStr++;
		#endif
	}
	
	#endif
	#ifndef NTAB
	printf("Hypervector: %d\n",num_initial);
	#ifdef DEBUG
	lineStr++;
	#endif
	
	for(int i=0; i<(num_initial); i++){
		printf("(HV {");
		for(int j=0; j<(*initial)[i].len; j++){
			printf("%d",(*initial)[i].vectors[j]);
			if(j!=((*initial)[i].len)-1) printf(",");
		}
		printf("}, %d, %d)\n",(*initial)[i].len, (*initial)[i].real?1:0);
		#ifdef DEBUG
		lineStr++;
		#endif
	}
	#endif
	#ifndef NO_INPUT
	printf("Press ENTER to start\n");
	
	getchar();
	#ifdef DEBUG
	lineStr+=2;
	#endif
	#endif
	
	#ifdef TIME
	begin = std::chrono::high_resolution_clock::now();
	#endif
	
	MatrixAdj = succinta(*Vertices, *Edges, *initial, num_vertices, num_edges, num_initial);
	
	#ifdef TIME
	end = std::chrono::high_resolution_clock::now();
	durataSucc = std::chrono::duration_cast<std::chrono::milliseconds>( end - begin).count();
	begin = std::chrono::high_resolution_clock::now();
	#endif
	
	#ifdef DEBUG
	#ifndef NTAB
	for(int i=0; i<num_initial; i++){
		for(int j=0; j<num_vertices && (*initial)[i].real; j++)
			printf("%d ",MatrixAdj[i][j]? 1:0);
		if((*initial)[i].real) {
			printf("\n");
			lineStr+=1;
		}
	}
	#endif
	#endif
	
	MatrixAdj = gpu_trans_succ(MatrixAdj, *initial, num_initial, num_vertices, num_edges);

	
	#ifdef TIME
	end = std::chrono::high_resolution_clock::now();
	durataClos = std::chrono::duration_cast<std::chrono::milliseconds>( end - begin).count();
	#endif
	
	printf("\n");
	#ifdef DEBUG
	lineStr+=2;
	#endif
	#ifndef NTAB
	for(int i=0; i<num_initial; i++){
		for(int j=0; j<num_vertices && (*initial)[i].real; j++)
			printf("%d ",MatrixAdj[i][j]? 1:0);
		if((*initial)[i].real) {
			printf("\n");
			#ifdef DEBUG
			lineStr+=1;
			#endif
		}
	}
	#endif
	
	#ifdef DEBUG
	printf("END\n");
	#ifdef DEBUG
	lineStr++;
	#endif
	#ifdef TIME 
	durata = durataRead+durataSucc+durataClos;
	printf("::durata: %lu ms\n",durata);
	#ifdef DEBUG
	lineStr++;
	#endif
	#endif
	#endif
	
	
	#ifdef FILE_OUT
	printf("Writing on file");
	if(argn>1){
		
		file_name = (file_name.substr(0,file_name.find(".", 2)) + ("_new.txt"));
		writeGraph(MatrixAdj,num_vertices, *initial, num_initial, num_edges, file_name );
	}else
		writeGraph(MatrixAdj,num_vertices, *initial, num_initial, num_edges, "prova_new.txt");
	#ifdef TIME
		for(int i=0; i<num_initial; i++){
			if((*initial)[i].real) {
				num_real_initial++;
			}
		}
		writeTime(num_vertices,num_edges_init, num_real_initial, num_edges, num_initial);
	#endif
	#endif
	printf("\nEND");
	
}
