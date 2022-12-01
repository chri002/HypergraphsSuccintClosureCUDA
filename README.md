# HypergraphsSuccintClosureCUDA
The succint closure on hypergraphs, with multi-threading on CUDA and openMP

## Calcolo della chiusura transitiva succinta dato un ipergrafo H, in maniera parallela ove se ne disponga le risorse.
<ul>
	<li>Parallelismo CPU: per ogni vertice del ipergrafo si invoca BFS; a seguire li merge dei vettori ottenuti dalla BFS. </li>
	<li>Parallelismo GPU: la BFS per trovare tutti i nipoti del nodo.</li>
	</ul>
	
## Calculation of the succinct transitive closure given a hypergraph H, in a parallel way where there is the Hardware.
<ul>
	<li>CPU parallelism: BFS is invoked for each vertex of the hypergraph; then it merge the vectors obtained from the BFS. </li>
	<li>GPU parallelism: find all the grandchildren of the node.</li>
	</ul>

## Installation:
it needs the NVIDIA toolkit to compile, and an NVIDIA graphics card to run.<br />
Copy all file in a folder, if you are using a windows PC run _compiler.bat_ to compile various combinations of executables, otherwise from the console run the commands as follows.

### Hypergraph generator
&emsp;\nvcc .\generator.cpp -o generator.exe <br />

### Succinct transitive closure of hypergraphs
&emsp;nvcc -rdc=true -lineinfo -std=c++17 <Flags> -Xcompiler -openmp .\progetto.cu -o progetto.exe <br /></br>
&emsp;Flags:</br>
&emsp;&emsp;-D HIDE  : hide the output of graph# <br />
&emsp;&emsp;-D DEBUG : show information on runtime <br />
&emsp;&emsp;-D FILE_OUT <value> : export graph to file <br />
&emsp;&emsp;-D MAX_THREADS <value> : max cuda threads <br />
&emsp;&emsp;-D MAX_BLOCKS_A <value> : max cuda blocks BFS <br />
&emsp;&emsp;-D MAX_BLOCKS_AI <value> : max cuda blocks BFS inside <br />
&emsp;&emsp;-D MAX_BLOCKS_B <value> : max cuda blocks succintion <br />
&emsp;&emsp;-D NTHR <value> : number of cpu threads <br />
&emsp;&emsp;-D TIME : enable time control <br />
&emsp;&emsp;-D NO_INPUT : remove enter click <br />
&emsp;&emsp;-D NTAB : hide the succinted graph outupt <br />
&emsp;&emsp;-D NO_DOUBLE : to use only external parallelization BFS CUDA  <br />
&emsp;&emsp;-D DYNAMIC : Automatic blocks number (Experimental)  <br />
&emsp;&emsp;-D CPU : custom BFS runs only on CPU, the other parts run on GPU yet <br />

## Work:
At begin generate the hypergraphs with _generator_ and then lunch _progetto_<br />

&emsp;&emsp;generator.exe -v &lt;number of vertices&gt; -e &lt;number of edges&gt; -s &lt;number of supersets&gt;<br />

&emsp;&emsp;progetto.exe "grafo.txt" <br />
