// Iterative procedure to learn field for repeat proteins:
// H(S) = - [  sum_i hi(ai) + sum_i sum_j J(a_i,a_j) -  lambda_id(S) ]: Agrego multiplicadores de Lagrange de pId
// To compile use
// g++ -std=gnu++11 -o LP_noLambda LP_noLambda.cpp
// Check inputs names on lines 63,71

#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <ctime>
#include <random>

using namespace std;
std::random_device rd;     // only used once to initialise (seed) engine
std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
std::uniform_int_distribution<int> uni20(0,20); // guaranteed unbiased
std::uniform_int_distribution<int> uni65(0,65); // guaranteed unbiased


int main()
{
	cout << "Here we go!" << endl;
	
	/* ----------------------------------------------------------------------------- */ 
	/* 1- LOAD DATA AND DEFINE PARAMETERS .......................................... */
	/* ----------------------------------------------------------------------------- */
	
	cout <<"  Loading data" <<endl;
	
	/* Parametros generales  | General parameters */
	int SavingDataSteps=20;           // Saving All Data every these N of Steps
	ofstream errorFile("Error.txt");
	
	const int naa=21;  // n of amino acids
	const int npos=66; // length sequences
	const int naanpos=naa*npos;  // columns/rows of the matrix Pij

	const double epsilonSingle=0.05; // epsilon to renew parameters hi
	const double epsilonJoint=0.01;   // epsilon to renew parameters jij
	const double gamma=0.001;
	
	const double maxErrorPermited=0.02;  // maximum error allowed between pij-fij and pi-fi
	const int ncomb=npos*(npos+1)/2-npos; // number of possible combinations of positions (1-1, 1-2, ... , 1-naa, 2-2,2-3,... etc)
	int PosPos[ncomb][2]; // all possible combinations of positions
	int a=0; //auxiliar counter
	for(int i=0;i<npos;i++){
		for(int j=i;j<npos;j++){
			if(i!=j){
				PosPos[a][0]=i;
				PosPos[a][1]=j;
				a+=1;
			}
		}
	}
	
	/* Read experimental frequencies data */
	double f1Data[naanpos];
	ifstream f1File("frecuenciasmarginales_train.txt"); 
	for (int j=0; j<naanpos; j++)
	{
		f1File >> f1Data[j]; 
	}
	f1File.close(); 

	static double f2Data[naanpos][naanpos];
	ifstream f2File("frecuenciasconjuntas_train.txt"); 
	for(int fila=0;fila<naanpos;fila++)
	{
		for(int col=0;col<naanpos;col++)
		{
			f2File >> f2Data[fila][col]; 
		}
	}
	f2File.close(); 

	/* ----------------------------------------------------------------------------- */
	/* 2- BEGINS BIG LOOP FOR PARAMETERS ACTUALIZATION ............................. */
	/* ----------------------------------------------------------------------------- */
	cout <<"Begins MC" <<endl;

	/* MC parameters */
	int FirstSeqSaved=1000; 	// first sequence I save in the MC procedure (to miss initial conditions)
	int SaveSequencesRate=1000;	// Every how many iterations should I consider the sequence
	int TotalSequences=80000; 	// How many sequences should I consider in the MC to calculate the Pij
	int TotalIterations=(TotalSequences+FirstSeqSaved)*SaveSequencesRate; // How many iterations should I do then
	
	/* Potts parameters matrices (Initially hi = log (fi) and Jij=0) */
	static double jij[naanpos][naanpos];
	for(int i=0;i<naanpos;i++){
		for(int j=0;j<naanpos;j++){
			jij[i][j]=0;
		}
	}
	static double hi[naanpos];
	for(int i=0;i<naanpos;i++){
		if(f1Data[i]!=0){
			hi[i]=log(f1Data[i]);
		}else{
			hi[i]=-10;
		}
	}

	// Definiciones de cosas a usar durante el while
	static double pij[naanpos][naanpos];
	static double p1[naanpos];
	double energy;

	double error=2.00;
	int counter=0;
	bool writeFiles=false;
	clock_t begin = clock();
	
	std::ofstream sequencesFile;
	std::ofstream energyFile;

	while(error > maxErrorPermited){
		counter+=1;
		
		writeFiles= (counter%SavingDataSteps==1);
		
		if(writeFiles){
			cout << counter << endl;
			//File to save sequences:
			std::string prefixSequenceFile="sequences";
			std::string fullnameSequences=prefixSequenceFile.append(std::to_string(counter));
			sequencesFile.open(fullnameSequences);
			
			//File to save sequences' energies
			std::string prefixEnergyFile="Energy";
			std::string fullnameEnergy=prefixEnergyFile.append(std::to_string(counter));
			energyFile.open(fullnameEnergy);
		}
		
		/* a) Recalculate Pij and Pi with the new parameters */
		// To do this I generate a large set of sequences with MC
		
		// reinicio valores de pij a cero
		for(int i=0;i<naanpos;i++){
			for(int j=0;j<naanpos;j++){
				pij[i][j]=0;
			}
		}
		
		//Random initial sequence:
		vector <int> sequence;
		for(int i=0; i<npos; i++)
		{
			int random_integer = uni20(rng);
			sequence.push_back(random_integer); 
		}
		
		// It's energy
		energy=0.00;
		for(int i=0; i<npos; i++)
		{
			energy+= -hi[naa*i+sequence[i]];
			if(i!=(npos-1)){
				for(int j=(i+1); j<npos; j++)
				{
					energy+= -jij[naa*(i)+sequence[i]][naa*(j)+sequence[j]];    
				}
			}
		}

		//   Do random mutations to recalculate Pij with new hi and jij parameters / Hago mutaciones y recorro el paisaje de secuencias para calcular los Pij
		int seqsavecounter=0;
		int sign;
		while(seqsavecounter<TotalIterations)
		{
			// Elijo mutacion al azar | random mutation
			int mpos= uni65(rng); 
			int mres= uni20(rng); 
			int oldRes=sequence[mpos];  

			// Calculo la variacion de energia con esta mutacion | change of energy
			double EiNewLocal= -hi[naa*mpos+mres]; 
			double EiNewInteraction=0.00; 
			double EiOldLocal= -hi[naa*mpos+oldRes]; 
			double EiOldInteraction=0.00; 

			for(int j=0; j<npos; j++)
			{
			  if(j!=mpos){
				EiNewInteraction+= -jij[ naa*(mpos)+mres][naa*(j)+sequence[j]]; 
				EiOldInteraction+= -jij[ naa*(mpos)+oldRes][naa*(j)+sequence[j]];
			  }
			}
			double dE= EiNewLocal+EiNewInteraction-EiOldLocal-EiOldInteraction; 

			// Elijo si me quedo o no con la mutacion | decide weater I keep the mutation
			if(dE < 0)
			{
				sequence[mpos]=mres;
				energy+=dE;
			}else{
				double r = ((double) rand() / (RAND_MAX)); 
				double expE=exp(-dE);                      
				if(r<expE){
					sequence[mpos]=mres;
					energy+=dE;

				}
			}
			seqsavecounter+=1;

			// Once every SaveSequenceRate, I refresh Pij values |
			// Renuevo los valores de Pij con los valores de una secuencia cada SaveSequencesRate
			int aux=seqsavecounter%SaveSequencesRate;  
			if(aux==1 & seqsavecounter>=(FirstSeqSaved*SaveSequencesRate))
			{  
				// Actualizo Pij con los valores de esta secuencia
				for(int j=0;j<ncomb; j++){
					int index1=PosPos[j][0]*naa+sequence[PosPos[j][0]];  
					int index2=PosPos[j][1]*naa+sequence[PosPos[j][1]];  
					pij[index1][index2]+=1;
					if(index1!=index2){
					  pij[index2][index1]+=1;
					}
				}
				if(sequencesFile.is_open()){
					// Write sequence and it's energy into files
					for (int j=0; j<npos; j++)
					{
						sequencesFile << sequence[j] << " ";
					}
					sequencesFile << endl;
					energyFile << energy << endl;
				}
			}
		}

		if(sequencesFile.is_open()){
			sequencesFile.close();
			energyFile.close();
		}

		// Normalization of Pij | Normalizo los Pij
		for(int index1=0;index1<naanpos; index1++){
			for(int index2=0;index2<naanpos;index2++){
				pij[index1][index2]=pij[index1][index2]/TotalSequences;
			}
		}
		
		// Calculate Pi values as partial sums of Pij | Calculo los nuevos Pi como suma parcial de Pij
		//reinicio valores de p1 a cero
		for(int indA=0;indA<naanpos;indA++){p1[indA]=0;}

		// For position 1 | Los de la posicion 1
		for(int indA=0;indA<naa;indA++){
			for(int indx=naa;indx<(2*naa);indx++){
				p1[indA]+=pij[indA][indx];
			}
		}
		// For other positions | Los de las demas posiciones
		for(int indA=naa;indA<naanpos;indA++){
			for(int indx=0;indx<naa;indx++){
				p1[indA]+=pij[indx][indA];
			}
		}

		/* b) Compare Pij and Pi (from the model) to fij and fi (from data) */
		double maxPij=0.00;  
		double m;  
		for(int fila=0;fila<naanpos;fila++)
		{
			int n1=fila/naa;
			for(int col=0;col<naanpos;col++)
			{
				int n2=col/naa;
				if(n1!=n2){
					m=abs(f2Data[fila][col] - pij[fila][col]);
					if(m>maxPij){maxPij=m;
					}
				}
			}
		}
		for(int i=0;i<naanpos;i++)
		{
			m=abs(f1Data[i]-p1[i]);
			if(m>maxPij){maxPij=m;
			}
		}
		error=maxPij;
		
		
		if(writeFiles){
			// Parameters hi and Jij before re-calculating them
			std::string prefixHI="ParamH";
			std::string fullnameHI=prefixHI.append(std::to_string(counter));
			ofstream hiFile(fullnameHI);
			for (int j=0; j<naanpos; j++)
			{
				hiFile << hi[j] << "\t";
				if((j+1)%naa==0){hiFile << endl;}
			}
			hiFile.close();
			cout <<"   ParamHI ";

			std::string prefixJij="ParamJ";
			std::string fullnameJIJ=prefixJij.append(std::to_string(counter));
			ofstream jijFile(fullnameJIJ);
			for(int fila=0;fila<naanpos;fila++)
			{
				for(int col=0;col<naanpos;col++)
				{
					jijFile << jij[fila][col] << "\t";
				}
				jijFile<< endl;
			}
			jijFile.close();
			cout <<"   ParamJij "<< endl;
			
			//Pij model distribution
			std::string prefixPIJN="Pij";
			std::string fullnamePIJN=prefixPIJN.append(std::to_string(counter));
			ofstream PijFile(fullnamePIJN);
			for(int fila=0;fila<naanpos;fila++)
			{
				for(int col=0;col<naanpos;col++)
				{
					PijFile << pij[fila][col] << "\t";
				}
				PijFile<< endl;
			}
			PijFile.close();
			cout <<"   PijDistrib ";
			
			//Pi model distribution
			std::string prefixPI="Pi";
			std::string fullnamePI=prefixPI.append(std::to_string(counter));
			ofstream PiFile(fullnamePI);
			for (int j=0; j<naanpos; j++)
			{
				PiFile << p1[j] << "\t";
				if((j+1)%naa==0){PiFile << endl;}
			}
			PiFile.close();
			cout <<"   PiDISTR ";
			
			errorFile << error << endl;
			
			// Mido cuanto tardo
			clock_t end = clock();
			double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			cout << "... Tiempo: " << elapsed_secs << endl;  
			clock_t begin = clock();
		}
		
		/* c) Redefine parameters values(hi and Jij), if error is still large */
		/* The Jij are defined according to L1 regularization */
		if(error > maxErrorPermited){
			// Change hi <- hi + epsilon log (f1 / p1)
			for(int i=0;i<naanpos;i++)
			{
				hi[i]+=epsilonSingle*(f1Data[i]-p1[i]);
			}
			// Change Jij <- Jij + epsilon log(f2 / pij)
			for(int fila=0;fila<naanpos;fila++)
			{	
				int n1=fila/naa;
				for(int col=0;col<naanpos;col++)
				{
					int n2=col/naa;
					if(n1!=n2){
					  
					  if(jij[fila][col]==0){
						if(abs(f2Data[fila][col]-pij[fila][col])<gamma){
						  jij[fila][col]=0;
						}else{
						  if(f2Data[fila][col]-pij[fila][col] <0){sign=-1;}else{sign=1;}
						  jij[fila][col]=+epsilonJoint*(f2Data[fila][col]-pij[fila][col]-gamma*sign);
						}
						
					    
					  }else{
						if(jij[fila][col]<0){sign=-1;}else{sign=1;}
						double auxJIJ=jij[fila][col]+epsilonJoint*(f2Data[fila][col]-pij[fila][col]-gamma*sign);
						if(auxJIJ*jij[fila][col]<0){jij[fila][col]=0;}else{jij[fila][col]=auxJIJ;}
						
					  }
					  
					  
					}
				}
			}
		}


	}

	/* ----------------------------------------------------------------------------- */
	/* 3- SAVE FINAL DATA .......................................................... */
	/* ----------------------------------------------------------------------------- */
	
	// Parameters hi and Jij before re-calculating them
	ofstream hiFinalFile("Final_hi.txt");
	for (int j=0; j<naanpos; j++)
	{
		hiFinalFile << hi[j] << "\t";
		if((j+1)%naa==0){hiFinalFile << endl;}
	}
	hiFinalFile.close();

	ofstream jijFinalFile("Final_jij.txt");
	for(int fila=0;fila<naanpos;fila++)
	{
		for(int col=0;col<naanpos;col++)
		{
			jijFinalFile << jij[fila][col] << "\t";
		}
		jijFinalFile<< endl;
	}
	jijFinalFile.close();
	
	//Pij model distribution
	ofstream PijFinalFile("Final_Pij.txt");
	for(int fila=0;fila<naanpos;fila++)
	{
		for(int col=0;col<naanpos;col++)
		{
			PijFinalFile << pij[fila][col] << "\t";
		}
		PijFinalFile<< endl;
	}
	PijFinalFile.close();
	
	//Pi model distribution
	ofstream PiFinalFile("Final_Pi.txt");
	for (int j=0; j<naanpos; j++)
	{
		PiFinalFile << p1[j] << "\t";
		if((j+1)%naa==0){PiFinalFile << endl;}
	}
	PiFinalFile.close();
	
	errorFile << error << endl;
	errorFile.close();
}


