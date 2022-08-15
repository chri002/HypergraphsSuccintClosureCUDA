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
	int n_v=25,m=0,n_e=n_v/2,n_s=n_v/3, to=0, m1,c, ms;
	std::string str="", strp="";
	
	for(int i=0;i<argn-1; i++){
		str = argvs[i];
		strp = argvs[i+1];
		if(str.substr(0,2)=="-v")
			n_v=std::atoi(argvs[i+1]);
		if(str.substr(0,2)=="-e")
			n_e=std::atoi(argvs[i+1]);
		if(str.substr(0,2)=="-s")
			n_s=std::atoi(argvs[i+1]);
	}
    
	const int NS = n_s;
	printf("Inizio creazione grafo (%d %d %d)\n", n_v, n_e, n_s);
	
	std::ofstream myFile;
	std::string name_file = "grafo_" + std::to_string(n_v) +"_"+ std::to_string(n_e) +".txt";
	myFile.open(name_file.c_str());
	myFile << "INI " << n_v <<"," << n_e <<"\n";
	
	int** superset= (int**) malloc(sizeof(int*)*n_s);
	void * supSet = malloc(sizeof(std::string)*n_s);
	bool ok=true;
	int i=0;
	
	while(i<n_s){
		ms = (std::rand()%(int)(sqrt(n_v)));
		m = 1+(std::rand()%(int)(n_v));
		m1=0,c=0;
		str="";
		for(int j=ms; j<m; j+=m1){
			str+= std::to_string(j);
			while(j+m1<=j){
				m1=(std::rand()%10);
			}
			c++;	
			if(j+m1<m) str+= ",";
			
		}
		ok=false;
		for(int j=0; j<i && !ok; j++){
			ok = ok || (((std::string*)supSet)[j]==str);
			
		}
		if(!ok && c>0){
			new (&((std::string*)supSet)[i]) std::string(str);
			superset[i] = (int*) malloc(sizeof(int)*(c+1));
			m=0; m1=fmax(str.find(","),m);
			
			for(int j=1;j<c+1; j++){
				superset[i][j] = std::stoi(str.substr(m,m1));
				m=m1+1;
				m1=str.find(",",m+1);
			}
			superset[i][0]=(c);
			
			printf("create superset %d: %s:%d\n",i,str.c_str(),c);
			i++;
		}
	}
	
	printf("write node\n");
	
	for(int i=0; i<n_v; i++){
		myFile <<  "(VE " << i <<")\n";
	}
	
	for(int i=0; i<n_e; i++){
		m = (std::rand()%(int)(n_s));
		int to = std::rand()%n_v;
		myFile << "(HA {";
		for(int j=1; j<=superset[m][0]; j+=1){
			myFile << superset[m][j];
			if(j<superset[m][0]) myFile <<  ",";
			
		}
		myFile << "}," << superset[m][0] << "," << to << ")\n";
		
		
		
	}
	myFile.close();
	
	
}