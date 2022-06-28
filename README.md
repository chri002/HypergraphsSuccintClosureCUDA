# HypergraphsSuccintClosureCUDA
The succint closure on hypergraphs, with multi-threading on CUDA and openMP

## Calcolo della chiusura transitiva succinta dato un ipergrafo H, in maniera parallela ove se ne disponga le risorse.
<ul>
	<li>Parallelismo CPU: per ogni vertice del ipergrafo si invoca BFS; merge dei vettori ottenuti dalla BFS. </li>
	<li>Parallelismo GPU: la BFS per trovare tutti i nipoti del nodo.</li>
	</ul>
	
## Calculation of the succinct transitive closure given a hypergraph H, in a parallel way where there is the Hardware.
<ul>
	<li>CPU parallelism: BFS is invoked for each vertex of the hypergraph; merge of vectors obtained from the BFS. </li>
	<li>GPU parallelism: the BFS to find all the grandchildren of the node.</li>
	</ul>

## Work:
progetto.exe "grafo.txt" <br />
(PRESS ENTER WHEN REQUEST)
