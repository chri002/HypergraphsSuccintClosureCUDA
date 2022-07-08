/*
#########################################################################################
#																						#
#	    	  			    hypergraph transitive closure 								#
#	nvcc -rdc=true -lineinfo -Xcompiler -openmp .\progetto.cu -o progetto.exe			#
#			 			-D HIDE  : hide the output										#
#			 			-D DEBUG : show information on runtime							#
#			 			-D FILE_OUT : export graph to file								#
#			 			-D MAX_THREADS : max cuda threads 								#
#			 			-D MAX_BLOCKS : max cuda blocks 								#
#			 			-D TIME : enable time control	 								#
#			 			-D NO_INPUT : remove enter clic 								#
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

Work:
	progetto.exe "grafo.txt"
	
Experimental:
	inside copyaa remove comment around std::copy and compile with -std c++17 
	end comment the openmp for cycle
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

//Massimo numero di threads su GPU e Block su GPU
//numero threads su CPU


#ifndef MAX_THREADS
#define MAX_THREADS 128
#endif
#ifndef MAX_BLOCKS
#define MAX_BLOCKS 1
#endif
#ifdef NTHR
	int nThr = NTHR;
#else
	int nThr;
#endif

#ifdef TIME
std::chrono::high_resolution_clock::time_point begin;
std::chrono::high_resolution_clock::time_point end;
unsigned long durata,durataRead;

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
	int* from;					//Insieme nodi provenienza CPU
	int* from_dev;				//Insieme nodi provenienza GPU
	int len_fr;					//lunghezza dei vettori precedenti
	int to;						//Nodo di arrivo dell'iperarco
} Hyperarc;

typedef struct{
	int* vectors;
	int len;
} Hypervector;

//Hyperarch comparison function for the unique function (NotEqual)
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
 
//Hyperarch comparison function for the unique function (EqualLess
bool compareTwoHyerarch(Hyperarc a, Hyperarc b)
{
	if(a.to!=b.to)
	return a.to < b.to;
	bool ok=true;
	int i,j=0;
	for(i=0; i<min(a.len_fr,b.len_fr) && ok; i++) {
		ok=ok && a.from[i]==b.from[i];
		j=i;
	}
	
	if(ok) return (a.len_fr<=b.len_fr);
	else return a.from[j]<b.from[j];
}

void printa(Hyperarc* hs, int l){
	printf("test %d\n",l);
	for(int i=0; i<l && hs[i].to!=-1; i++){
		printf("(HA {");
		for(int j=0; j<(hs)[i].len_fr; j++){
			printf("%d",(hs)[i].from[j]);
			if(j!=((hs)[i].len_fr)-1) printf(",");
		}
		printf("},%d,%d)\n",(hs)[i].len_fr,(hs)[i].to);
	}
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
!!!!!!!
MODIFY:
	**Vertices, **Edges, num_edges, num_vertices
*/
void readGraph(std::string FILE, int** Vertices, int &num_vertices, Hyperarc ** Edges, int &num_edges, Hypervector** initial, int &num_initial){
	std::ifstream file_graph;
	std::string line, pref, from;
	int idxE = 0, idxV = 0, len_fr, to, temp, temp1,temp2;
	std::map<std::string, int> temporaneo;
	
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
				for(int i=0; i<len_fr; i++){
					temp1 = from.find(",", temp);
					if(temp==-1)
						(*Edges)[idxE].from[i] = std::stoi(from.substr(temp));
					else
						(*Edges)[idxE].from[i] = std::stoi(from.substr(temp,temp1));
					temp=temp1+1;
					
				}
				if(temporaneo.find(from)==temporaneo.end())
					temporaneo.insert(make_pair(from,len_fr));
				idxE++;
			}else if(pref == "(VE"){
				(*Vertices)[idxV] = std::stoi(line.substr(3, line.find(")")-1));
				idxV ++;
			}
		}
		*initial = (Hypervector*) malloc(sizeof(Hypervector*)*temporaneo.size());
		int itera=0;
		
		for(auto it=temporaneo.begin(); it!=temporaneo.end(); it++){
			temp=0; temp1=0;
			(*initial)[itera] = {(int*)malloc(sizeof(int)*it->second),it->second};
			for(int j=0; j<it->second; j++){
				temp1=it->first.find(",", temp);
				(*initial)[itera].vectors[j] =  std::stoi(it->first.substr(temp,temp1-temp));
				temp = temp1+1;
			}
			
			itera++;
		}
			
		num_initial = temporaneo.size();
		file_graph.close();
	}
	
}

/*   ## CPU ##
Write graph from file
*/
/*
Input:
	Vertices : pointer of integers 
	num_vertices : number of integers pointed from Vertices
	Edges : pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from Edges
	FILE : relative path and file name
*/
void writeGraph(int* Vertices, int num_vertices, Hyperarc* Edges, int num_edges, std::string FILE){
	std::ofstream myFile;
	myFile.open(FILE);
	myFile << "INI " << num_vertices <<"," << num_edges <<"\n";
	for(int i=0; i<num_vertices; i++)
		myFile << "(VE " << i <<")\n";
	for(int i=0; i<(num_edges); i++){
		myFile << "(HA {";
		for(int j=0; j<(Edges)[i].len_fr; j++){
			myFile << (Edges)[i].from[j];
			if(j!=((Edges)[i].len_fr)-1) myFile << ",";
		}
		myFile << "}," << (Edges)[i].len_fr<<","<<(Edges)[i].to <<")\n";
			
	}
	
	
	
	myFile.close();
}	

/*   ## CPU ##
Write operation time to file
*/
/*
Input:
	num_vertices : number of integers pointed from Vertices
	num_edges : number of hyperarcs pointed from Edges
*/
void writeTime(int num_vertices, int num_edges){
	std::ofstream myFile;
	myFile.open("TimeSave.txt", std::fstream::app);
	myFile << "<G " << num_vertices <<"," << num_edges <<" ("<<durata<<" ns)>\n";
	myFile.close();
}


__device__ bool equalHyperVector(int * hyperarcs, int * vectors, int minLen){
	bool ok=true;
	for(int i=0; i<minLen && ok; i++){
		ok=hyperarcs[i]==vectors[i];
	}
	return ok;
}


/*   ## GPU ##
Find neighbors during one BFS step
*/
/*
Input:
	Vertices : pointer of integers 
	num_vertices : number of integers pointed from Vertices
	Edges : pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from Edges
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
	thidLast : the node from to which to search for hyperarcs
!!!!!!
MODIFY:
	Cost, Visited
*/
__global__ void neighOp(int *Vertices, int num_vertices, Hyperarc * Edges, int num_edges, bool* FrontierUpdate, bool* Visited, int* Cost, Hypervector thidLast, int first){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	bool thereIs = false;
	int thidI;
	
	
	for(int Pass=0; Pass<ceilf((num_edges/(blockDim.x)))+2; Pass++){
		thidI = thid + Pass*blockDim.x;
	
		if(thidI<num_edges){
			for(int i=0; i<num_vertices; i++){
				thereIs = false;	
				if(Edges[thidI].to==Vertices[i]){
										
						if(equalHyperVector(Edges[thidI].from_dev,thidLast.vectors,thidLast.len) && first==1){
							if(Visited[Vertices[i]]==false){
								Cost[Vertices[i]] = Cost[thidLast.vectors[0]]+1;
								FrontierUpdate[Vertices[i]] = true;
							}
						}else if(first!=1){
							for(int j=0; j<Edges[thidI].len_fr && !thereIs; j++){
								thereIs = Edges[thidI].from_dev[j]==thidLast.vectors[0];
							}
							if(thereIs){
								if(Visited[Vertices[i]]==false){
									Cost[Vertices[i]] = Cost[thidLast.vectors[0]]+1;
									FrontierUpdate[Vertices[i]] = true;
								}
							}
						}
					
				}
			}
		}
	}
	__syncthreads();

}


/*   ## GPU ##
One step of BFS 
*/
/*
Input:
	Vertices : pointer of integers 
	num_vertices : number of integers pointed from Vertices
	Edges : pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from Edges
	Frontier : the Frontier of the step
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
!!!!!!
MODIFY:
	Frontier, Cost, Visited
!!!!!!
Kernel invoked (1 GPU block)
*/	
__global__ void bfs(int *Vertices, int num_vertices, Hyperarc * Edges,const int num_edges, bool * Frontier, bool* FrontierUpdate, bool* Visited, int* Cost, int* first, int* len){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int thidI, idx=0;
	
	if(*len!=-1){
		if(thid==0)
			neighOp<<< 1,min(num_edges,MAX_THREADS)>>>(Vertices, num_vertices,  Edges, num_edges, FrontierUpdate, Visited, Cost, {first,*len}, 1);
	}else
		for(int Pass=0; Pass<ceilf(num_vertices/(blockDim.x))+2; Pass++){
			thidI = thid + Pass*(blockDim.x);
			if(thidI < num_vertices && Frontier[thidI]){
				Frontier[thidI] = false;
				neighOp<<< 1,min(num_edges,MAX_THREADS)>>>(Vertices, num_vertices,  Edges, num_edges, FrontierUpdate, Visited, Cost, {new int[]{thid},1}, 0);
			}
			
		}
	
	__syncthreads();
	
}


/*   ## GPU ##
Update the ausiliary structur of BFS
*/
/*
Input:
	num_vertices : number of integers pointed from Vertices
	FrontierUpdate : tracks changes during BFS step
	Visited : tracks the visited nodes during BFS
	Cost : the distance between start node and the other node
	next : check wheater to proceed or not
!!!!!!
MODIFY:
	Frontier, Visited, next, FrontierUpdate
*/
__global__ void bfs_update(int num_vertices, bool * Frontier, bool * FrontierUpdate, bool * Visited, int * next){
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int thidI;
	for(int Pass=0; Pass<ceilf((num_vertices/(blockDim.x)))+2; Pass++){
		thidI = thid + Pass*(blockDim.x );
		if(thidI<num_vertices && FrontierUpdate[thidI]){
			Frontier[thidI] = true;
			Visited[thidI] = true;
			*next = 1;
			FrontierUpdate[thidI] = false;
			
		}
	}
	__syncthreads();
	
}


/*   ## CPU ##
Calculate the neighbors through BFS on GPU
*/
/*
Input:
	Vertices : pointer of integers 
	num_vertices : number of integers pointed from Vertices
	Edges : pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from Edges
	node : the starting node of BFS
Return:
	Succint Closure hyperarcs
!!!!!!
MODIFY:
	Edges
!!!!!!
Sequentially
	Kernel invoked (MAX_BLOCKS GPU block)
	Kernel invoked (MAX_BLOCKS GPU block)
!!!!!!
Allocate num_vertices*3 bool array CPU/GPU
Allocate num_vertices*1 int  array CPU
Allocate num_vertices*2 int  array GPU
Allocate num_edges Hyperarc  array GPU
Allocate new Edges array that contains the succint closure hyperarcs
Allocate Temporary Array to copy to GPU the set of node of the hyperarcs

*/
Hyperarc * Edges_DEV;
int * Vertices_DEV;

Hyperarc * graph_bfs_nieces(int * Vertices, int num_vertices, Hyperarc * Edges, const int num_edges, Hypervector node){
	int * Cost_HOS, *Cost_DEV;
	bool * Frontier_HOS, *Frontier_DEV;
	bool * FrontierUpdate_HOS, *FrontierUpdate_DEV;
	bool * Visited_HOS, *Visited_DEV;
	
	int * from;
	Hyperarc *newEdges;
	int sizeEdges=0;
	
	Cost_HOS = (int*) malloc(sizeof(int)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Cost_DEV, sizeof(int)*num_vertices));
	
	Frontier_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Frontier_DEV, sizeof(bool)*num_vertices));
	
	FrontierUpdate_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&FrontierUpdate_DEV, sizeof(bool)*num_vertices));
	
	Visited_HOS = (bool*) malloc(sizeof(bool)*num_vertices);
	gpuErrchk(cudaMalloc((void**)&Visited_DEV, sizeof(bool)*num_vertices));
	
	printf("ORA 0\n");
	
	for(int i=0; i<num_vertices; i++){
		Cost_HOS[i] = -1;
		Frontier_HOS[i] = false;
		FrontierUpdate_HOS[i] = false;
		Visited_HOS[i] = false;
	}
	for(int i=0; i<node.len; i++){
		Frontier_HOS[node.vectors[i]] 	= true;
		Visited_HOS[node.vectors[i]] 	= true;
		Cost_HOS[node.vectors[i]] 		= 0;
	}
	
	cudaMemcpy(Cost_DEV, Cost_HOS, sizeof(int)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(Frontier_DEV, Frontier_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(FrontierUpdate_DEV, FrontierUpdate_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	cudaMemcpy(Visited_DEV, Visited_HOS, sizeof(bool)*num_vertices, cudaMemcpyHostToDevice);
	
	
	int *next_HOS, *next_DEV, *len_HOS, *len_DEV;
	
	gpuErrchk(cudaMalloc((void**) &next_DEV, sizeof(int)));
	gpuErrchk(cudaMalloc((void**) &len_DEV, sizeof(int)));
	
	
	
	
	next_HOS = (int*) malloc(sizeof(int));
	len_HOS = (int*) malloc(sizeof(int));
	
	gpuErrchk(cudaMalloc((void**)&from, sizeof(int)*node.len));
			
	gpuErrchk(cudaMemcpy(from, node.vectors,  sizeof(int)*node.len, cudaMemcpyHostToDevice));
	
	*len_HOS = node.len;
	gpuErrchk(cudaMemcpy(len_DEV, len_HOS,  sizeof(int), cudaMemcpyHostToDevice));
	

	
	*next_HOS = 1;
	
	
	while(*next_HOS==1){
		*next_HOS = 0;
		
		gpuErrchk(cudaMemcpy(next_DEV, next_HOS, sizeof(int), cudaMemcpyHostToDevice));
		
		bfs<<<MAX_BLOCKS, min(num_vertices, MAX_THREADS) >>>(Vertices_DEV, num_vertices, Edges_DEV, num_edges, Frontier_DEV, FrontierUpdate_DEV, Visited_DEV, Cost_DEV, from, len_DEV);
		
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();
		
		bfs_update<<<MAX_BLOCKS, min(num_vertices, MAX_THREADS) >>>(num_vertices, Frontier_DEV, FrontierUpdate_DEV, Visited_DEV, next_DEV);
		gpuErrchk( cudaPeekAtLastError() );
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy(next_HOS, next_DEV , sizeof(int), cudaMemcpyDeviceToHost));
		
		*len_HOS = -1;
		
		gpuErrchk(cudaMemcpy(len_DEV, len_HOS, sizeof(int), cudaMemcpyHostToDevice));
		

		
	}
	
	
	gpuErrchk(cudaFree(from));
	gpuErrchk(cudaFree(next_DEV));
	gpuErrchk(cudaFree(len_DEV));
	gpuErrchk(cudaFree(Frontier_DEV));
	gpuErrchk(cudaFree(FrontierUpdate_DEV));
	gpuErrchk(cudaFree(Visited_DEV));
	
	(free(Frontier_HOS));
	(free(FrontierUpdate_HOS));
	(free(Visited_HOS));
	
	cudaMemcpy(Cost_HOS, Cost_DEV, sizeof(int)*num_vertices, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaFree(Cost_DEV));
	
	for(int i=0; i<num_vertices; i++){
		if(Cost_HOS[Vertices[i]]>1) sizeEdges++;
	}
	
	
	
	newEdges = (Hyperarc*) malloc(sizeof(Hyperarc)*(sizeEdges+1));
	int k=0;
	for(int i=0; i<num_vertices && k<sizeEdges; i++){
		if(Cost_HOS[Vertices[i]]>1){
			newEdges[k] = {(int*)malloc(sizeof(int)*node.len),NULL,node.len,Vertices[i]};
			std::copy(node.vectors,(node.vectors)+(node.len), newEdges[k].from);
						
			gpuErrchk(cudaMalloc((void**)&from, sizeof(int)*node.len));
			gpuErrchk( cudaPeekAtLastError() );
		
			cudaMemcpy(from, node.vectors, sizeof(int), cudaMemcpyHostToDevice);
			gpuErrchk( cudaPeekAtLastError() );
		
			newEdges[k].from_dev = from;
						
			k++;
		}
	}
	
	
	newEdges[k] = {NULL,NULL,-1,-1};
	
	return newEdges;
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
!!!!!!
MULTI PARALLELISM OPENMP
*/
int copyaa(Hyperarc* a, Hyperarc* b, int w){
	int i;
	/*
	#pragma omp parallel for private(i) shared(a,b) num_threads(nThr)
	for(i=0; i<w; i++){
		a[i] = b[i];
	}
	*/
	//std::memcpy(a,b,w*sizeof(Hyperarc));
	
	//using std c++17
	std::copy(std::execution::par,b,b+w,a);
	
	std::sort(a, a + w, compareTwoHyerarch);
	std::unique( a, a + w , ne_compareTwoHyerarch);
	return 0;
}



/*   ## CPU ##
Prepare the data and lunch BFS
*/
/*
Input:
	Vertices : pointer of integers 
	num_vertices : number of integers pointed from Vertices
	Edges : pointer of pointer of Hyperarcs
	num_edges : number of hyperarcs pointed from Edges
	Start : pointer of Integers (the vertices from which to start the BFS)
	num_start : number of integers pointed from Start
!!!!!!
MODIFY:
	Edges
!!!!!!
MULTI PARALLELISM OPENMP
!!!!!!
Allocate (num_start)* int array CPU (split between threads)

*/
void gpu_bfs(int * Vertices, int num_vertices, Hyperarc ** Edges, int &num_edges, Hypervector * Start, int num_start){
	Hyperarc ** nArch, * newEdges;
	int nNew_Arch=0, *sizeTot;
	int * from;
	
	

	gpuErrchk(cudaMalloc((void**)&Vertices_DEV, sizeof(int)*num_vertices));
	gpuErrchk(cudaMalloc((void**)&Edges_DEV, sizeof(Hyperarc)*num_edges));
	
	cudaMemcpy(Vertices_DEV, Vertices, sizeof(int)*num_vertices, cudaMemcpyHostToDevice);
	int len_frT = 0;
	for(int i=0; i<num_edges; i++){	
		len_frT = (*Edges)[i].len_fr;
		gpuErrchk(cudaMalloc((void**)&from, sizeof(int) * len_frT));
		
		cudaMemcpy(from, (*Edges)[i].from, sizeof(int)* len_frT, cudaMemcpyHostToDevice);
		
		
		(*Edges)[i].from_dev = from;
		
	}
	
	cudaMemcpy(Edges_DEV, *Edges, sizeof(Hyperarc)*num_edges, cudaMemcpyHostToDevice);
	
	
	
	#pragma omp parallel num_threads(nThr) shared(Vertices, Edges, Start, num_vertices, num_edges, num_start, nNew_Arch, sizeTot, newEdges) private(nArch) 
	{
		if(omp_get_thread_num()==0) sizeTot = (int*) malloc(sizeof(int)*omp_get_num_threads());
		
		int workN = ceil((double)num_start/(double)(omp_get_num_threads()));
		int ini = workN * omp_get_thread_num();
		int end = fmin(workN * omp_get_thread_num() + workN, num_vertices);
		int * mySyze = (int*)malloc(sizeof(int)*(end-ini));
		int mySSyze=0;
		
	
		
		#ifdef DEBUG
		printf("%d (%d,%d)\n",omp_get_thread_num(),ini,end);
		#endif
		
		nArch = (Hyperarc**)malloc(sizeof(Hyperarc*)*workN);
		
		for(int i=ini; i<end; i++){
			mySyze[i-ini]=0;
			nArch[i-ini] = graph_bfs_nieces(Vertices, num_vertices, *Edges, num_edges, Start[i]);
			
			for(int j=0; j<num_vertices && (nArch[i-ini][j].to!=-1); j++){
				#pragma omp atomic
				nNew_Arch+=1;
				
				mySSyze++;
				mySyze[i-ini]++;
			}
		}
		#ifdef DEBUG
		printf("%d ",omp_get_thread_num());
		#endif
		
		sizeTot[omp_get_thread_num()] = mySSyze;
		
		#pragma omp barrier
		if(omp_get_thread_num()==0){
			newEdges = (Hyperarc*) malloc(sizeof(Hyperarc)*(nNew_Arch+num_edges));
			#ifdef DEBUG
			printf("\n");
			#endif
		}
		
		#pragma omp barrier
		int idx=0,ida,len=end-ini;
		
		ini=0;
		for(int i=0; i<omp_get_thread_num(); i++) ini+=sizeTot[i];
		
		#ifdef DEBUG
		printf("%d: ini %d-%d (%d)\n", omp_get_thread_num(), ini, ini+mySSyze, (nNew_Arch+num_edges));
		#endif
		
		ida=ini; 
		
		while(ida<ini+mySSyze){
			for(int j=0;j<len; j++){
				#pragma omp critical
				{
					if(idx<mySyze[j]){
						newEdges[ida] = nArch[j][idx];
						ida++;
						
					
					}
				}
				
			}
			idx++;
			
		}
		#ifdef DEBUG
		printf("finito %d\n",omp_get_thread_num());
		#endif
		#pragma omp barrier
	}
	
	gpuErrchk(cudaFree(Edges_DEV));
	gpuErrchk(cudaFree(Vertices_DEV));
	
	#ifdef DEBUG
	printf("copia finale\n");
	#endif
	
	for(int i=0; i<num_edges; i++)
		newEdges[i+nNew_Arch] = (*Edges)[i];
	
	num_edges += nNew_Arch;
	free(*Edges);
	
	
	
	printf("inizio copia");
	(*Edges) = (Hyperarc*) malloc(sizeof(Hyperarc*)*num_edges);
	(*Edges) = newEdges;
	printf(" %s",*Edges==NULL? "error":"");
	printf("%s\n",copyaa((*Edges),(newEdges), (num_edges))==0?"terminata":" error");
	
	
	
}


/*   ## CPU ##
main function
read graph and initialize it, lunch gpu_bfs, write graphs
(writing on console or file depends on the type of compilation)
*/
/*
Input:
	args = { name_program, [name_graphs] }

*/
int main(int argn, char ** args){
	
	
	
	
	system("cls");
	
	int ** Vertices /*= (int*) malloc(sizeof(int)*8);*/ = (int**) malloc(sizeof(int*));
	int num_vertices=0, num_edges=0, num_initial=0;
	bool argg=false;
	std::string file_name;
	Hyperarc ** Edges  = (Hyperarc**) malloc(sizeof(Hyperarc*));
	Hypervector ** initial = (Hypervector**) malloc(sizeof(Hypervector*));
	
	if(argn>1){
		for(int i=0; i<argn; i++)
			if(std::string(args[i]).substr(0,4) == "-nT="){
				nThr = std::stoi(std::string(args[i]).substr(4));
				argg=true;
			}
			else 
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
		readGraph("prova.txt",Vertices,num_vertices, Edges, num_edges, initial, num_initial);
		
		#ifdef TIME
		end = std::chrono::high_resolution_clock::now();
		#endif
	}
	
	
	
	#ifdef TIME
	durataRead = std::chrono::duration_cast<std::chrono::nanoseconds>( end - begin ).count();
	#ifdef DEBUG
	printf("reading time: %lu ns\n",durataRead);
	printf("CPU threads: %d\nGPU Block:%d\nGPU threads: %d\n", nThr,MAX_BLOCKS,MAX_THREADS);
	#endif
	#endif
	
	#ifndef HIDE
	for(int i=0; i<(num_edges); i++){
		printf("(HA {");
		for(int j=0; j<(*Edges)[i].len_fr; j++){
			printf("%d",(*Edges)[i].from[j]);
			if(j!=((*Edges)[i].len_fr)-1) printf(",");
		}
		printf("} %d)\n",(*Edges)[i].to);
			
	}
	
	for(int i=0; i<num_vertices; i++)
		printf("(VE %d)\n",(*Vertices)[i]);
	
	for(int i=0; i<(num_initial); i++){
		printf("(HV {");
		for(int j=0; j<(*initial)[i].len; j++){
			printf("%d",(*initial)[i].vectors[j]);
			if(j!=((*initial)[i].len)-1) printf(",");
		}
		printf("})\n");
			
	}
	#endif
	
	#ifndef NO_INPUT
	printf("Press ENTER to start\n");
	
	getchar();
	#endif
	#ifdef TIME
	begin = std::chrono::high_resolution_clock::now();
	#endif
	gpu_bfs(*Vertices, num_vertices, Edges, num_edges, *initial, num_initial);

	#ifdef TIME
	end = std::chrono::high_resolution_clock::now();
	durata = std::chrono::duration_cast<std::chrono::nanoseconds>( end - begin).count();
	#endif
	
	
	
	#ifdef DEBUG
	printf("END\n");
	#ifdef TIME 
	printf("::durata: %lu ns\n",durata);
	#endif
	#endif
	#ifndef HIDE
	
	for(int i=0; i<(num_edges); i++){
		printf("(HA {");
		for(int j=0; j<(*Edges)[i].len_fr; j++){
			printf("%d",(*Edges)[i].from[j]);
			if(j!=((*Edges)[i].len_fr)-1) printf(",");
		}
		printf("},%d,%d)\n",(*Edges)[i].len_fr,(*Edges)[i].to);
	}
	#endif
	
	#ifdef FILE_OUT
	if(argn>1){
		
		file_name = (file_name.substr(0,file_name.find(".", 2)) + ("_new.txt"));
		writeGraph(*Vertices,num_vertices, *Edges, num_edges, file_name );
	}else
		writeGraph(*Vertices,num_vertices, *Edges, num_edges,"prova_new.txt");
	#ifdef TIME
		writeTime(num_vertices,num_edges);
	#endif
	#endif
	
}
	
	
