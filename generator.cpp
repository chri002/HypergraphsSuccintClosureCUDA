/*
#############################################################
#															#
#				  hypergraph generator 						#
#			nvcc .\generator.cpp -o generator.exe			#
#			 cl .\generator.cpp -o generator.exe			#
#															#
#############################################################
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <iostream>
#include <fstream>


int main(int argn, char ** argvs){
	std::srand(std::time(NULL)); // use current time as seed for random generator
    int n = argn>1? std::stoi(argvs[1]): 25, m=0, n2=(argn>2? std::stoi(argvs[2]):n/2);
	
	std::ofstream myFile;
	myFile.open("grafo.txt");
	myFile << "INI " << n <<"," << n2 <<"\n";
	
	for(int i=0; i<fmax(n,n2); i++){
		m = 1+(std::rand()%(int)(n));
		if(i<(int)n2)
		{
			myFile << "(HA {";
			int m1=0,c=0;
			for(int j=0; j<m && c<n/1.2; j+=m1){
				myFile << j;
				while(j+m1<=j){
					m1=(std::rand()%10);
				}
				c++;	
				if(j+m1<m && c<n/1.2) myFile <<  ",";
				
			}
			myFile << "}," << c << "," << std::rand()%n << ")\n";
		}
		if(i<n) myFile <<  "(VE " << i <<")\n";
		
	}
	myFile.close();
	
	
}