// Divide and Conquer Algorithm Implementation :
//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>



using namespace std;

//const int N = 500;
double start_time;
double finish_time;
double gather_start;
double gather_finish;
double bcast_start;
double bcast_finish;


//Generate random numbers between fMin and fMax
double fRand(double fMin, double fMax)
{
   double f = (double)rand() / RAND_MAX;
   return fMin + f * (fMax - fMin);
}

//Multiply a matrix represented by 3 vectors by a vector
void multiply_vec(vector<double> a, vector<double> b, vector <double> c, vector <double> d,vector <double> &sol, int N){
  for (int i = 0; i < N; i++){
    if (i ==0){
      sol[i] = b[i]*d[i] + c[i]*d[i+1];
    }
    else if (i == N-1){
      sol[i] = a[i-1]*d[i-1] + b[i]*d[i];
    }
    else{
      sol[i] = a[i-1]*d[i-1] + b[i]*d[i] + c[i]*d[i+1]; 
    }
  }
}

//Generate the tridiagonal matrix represented by 3 vectors 
void vec_generator(vector<double> &a, vector <double> &b, vector <double> &c, vector <double> &sol, int N){
  srand(time(NULL));

  //Create diagnoally dominate set of 3 vectors representing the matrix
  for (int i = 0; i < N; i++){
    if(i ==0){
      b[i] = fRand(0,100);
      c[i] = fRand(0,b[i]);
    }
    else if (i == N-1){
      b[i] = fRand(0,100);
      a[i-1] = fRand(0,b[i]);
    }
    else{
      a[i-1] = fRand(0,100);
      c[i] = fRand(0,100);
      b[i] = fRand(a[i-1]+c[i],200);
    }
  }

  //find solution vector
  vector <double> d;

  for (int i = 0; i <N; i ++){
    d.push_back(i+1.0);
  }

  multiply_vec(a,b,c,d,sol,N);
  
}


//Function to print a single vector
void print_a_vector(vector <double> a, int N){
  for (int i = 0; i < N; i++){
    cout << a[i] << ", ";
  }
  cout << endl;
}


//Print 3 vectors as a matrix
void print_vec(vector <double> a,vector  <double> b, vector<double>  c, int N){
  int index = -1;
  for (int i =0; i < N; i++){
    for (int j =0; j<N;j++){
      
      if (j == 0 && index == -1){
	cout << b[i] << ",  " << c[i];
	j += 1;
      }
      else if (i == N-1 && index == j) {
	cout << a[i-1] << ",  " << b[i];
	
      }
      else if(index == j){
	cout << a[i-1] << ",  " << b[i] << ",  " << c[i];
	j += 2;
      }
      else {
	cout <<  "0,  ";
      }
      
    }
    cout << endl;
    index++;
  }
}

//Serial Thomas Algorithm implementation
vector <double> thomas_algorithm(vector<double> a,
				 vector<double> b,
				 vector<double> c,
				 vector <double> d,
				 int size) {

  double m;
  //forward elimination phase
  for (int i = 1; i < size; i++){
    m = a[i-1]/b[i-1];
    b[i] = b[i] - m*c[i-1];
    d[i] = d[i] - m*d[i-1];
  }
  
  //backward substitution phase
  d[size-1] = d[size-1]/b[size-1];
  for (int i = size-2; i >= 0; i--){
    d[i] = (d[i]-c[i]*d[i+1])/b[i];
  }
  return d;
    
}



int main (int argc, char *argv[])
{
 
        MPI::Init(argc,argv);                       //   Initialize MPI
	MPI::Comm & comm = MPI::COMM_WORLD;         //

	int comm_sz = comm.Get_size();              //   Total number of processors
	int my_rank = comm.Get_rank();              //   Rank of current processor

	
	//Get the desired problem size
	int N = atoi(argv[1]);                      //   Problem Size N
	const int k = N/comm_sz;                    //   k-value


       

	     
	vector<double> y(k);               //
	vector<double> ek(k);              //
	vector<double> e1(k);              // All used to calulculate local variables y and zm/zm1 on each processor
	vector<double> zm(k);              //
	vector<double> zm1(k);             //




	vector <double> a(N);              //
	vector <double> b(N);              // Matrix element vectors
	vector <double> c(N);              //
	vector <double> sol(N);            //

	vector <double> a1(k);             //
	vector <double> b1(k);             // Local matrix element vectors
	vector <double> c1(k);             //
	vector <double> d1(k);             //

	vector <double> package;           // Used to scatter a,b,c, and sol as a single message to reduce overhead
	package.reserve(4*N);              // Allocate memory to the package

	vector<double> recieve1(4*k);      // Used to recieve initial scaller

	vector <double> package2;          //
	package.reserve(8);                // Used to gather local reduced matrix solutions
	vector<double> recieve(8*comm_sz); //


	vector<double> s(2*comm_sz-2);     //
	vector<double> r(2*comm_sz-2);     // Vectors of the reconstructed Matrix of size 2*(# of processors)-2
	vector<double> t(2*comm_sz-2);     //
	vector<double> u(2*comm_sz-2);     //
	



	//initialize E1 and Ek
	e1[0] = 1;
	ek[k-1] = 1;

	
	int f = 0;
	//Processor 0 creates tridiagonal matrix 
	if (my_rank == 0){
	  
	  vec_generator(a,b,c,sol,N);
	  

	  //Partition the problem for each processor
	  
	  for(int j = 0; j < comm_sz; j++){
	    f = 0;
	    for (int i = j*k; i < (j*k)+k; i++){
	      d1[f] = sol[i];
	      b1[f] = b[i];
	      if (i != (j*k)+k-1){
		a1[f] = a[i];
		c1[f] = c[i];
	      }
	      else{
		a1[f] = a[i];
		if (i < k){
		  c1[f] = 0;
		}
		else{
		  c1[f] = c[i-k];
		}
		
	      }
	      f++;
	     }
	    //Add the processor partition just created to the scatter package
	    package.insert(package.end(),a1.begin(),a1.end());                   //
	    package.insert(package.end(),b1.begin(),b1.end());                   //  Combine vectors into one vector before the scatter 
	    package.insert(package.end(),c1.begin(),c1.end());                   //  
	    package.insert(package.end(),d1.begin(),d1.end());                   //
	   }

	 }


	//Scatter information amongst Processors
	comm.Scatter(&package.front(),4*k,MPI_DOUBLE,&recieve1.front(),4*k,MPI_DOUBLE,0);                    //  Now requires only one scatter


	
	//Unpack the vectors from the container after scatter
	double a_const = 0;
	double c_const = 0;
	f = 0;
	for (int i =0; i < 4*k; i++){
	    
	  
	    if (i < k){ 
	      if (f == k-1) {
		a_const = recieve1[i];
		a1[f] = 0;
	      }
	      else{
	        a1[f] = recieve1[i];
	      }
	    }
	    else if (i < 2*k){
	      b1[f] = recieve1[i];
	      
	    }
	    else if (i < 3*k){
	      if (f == k-1) {
		c_const = recieve1[i];
		c1[f] = 0;
	      }
	      else{
	        c1[f] = recieve1[i];
	      }	     
	    }
	    else {
	      d1[f] = recieve1[i];
	      
	    }
	    f++;
	    if (f == k){ f = 0;}
	    
	}
	
	/*Code to check Scatter
	if(my_rank ==0){
	  cout << "after scatter: " << endl;
	  print_a_vector(recieve1,4*k);
	  cout << endl << endl;
	  print_a_vector(a1,k);
	  print_a_vector(b1,k);
	  print_a_vector(c1,k);
	  cout << "a_const: " << a_const << "   c_const: " << c_const << endl;
	  }*/
	                                                                    
	//Start timing soluton computation once problem is on each processor
	start_time = MPI::Wtime();
	
       
	//Solve local systems on each processor

	y = thomas_algorithm(a1,b1,c1,d1,k);
	if (my_rank == 0){  
	  zm = thomas_algorithm(a1,b1,c1,ek,k);
	}
	else if (my_rank == comm_sz-1){
	  zm = thomas_algorithm(a1,b1,c1,e1,k);
	}
	else{
	  zm = thomas_algorithm(a1,b1,c1,e1,k);
	  zm1 = thomas_algorithm(a1,b1,c1,ek,k);
	}

	

	/*Code to check solution on individual processor
	if(my_rank != 0){
	  cout << "solution on p " << my_rank << "is: " << endl;
	  print_a_vector(y,k);

	  cout << "Zm is equal to: " << endl;
	  print_a_vector(zm,k);
	  
	  cout << "zm1 is equal to: " << endl;
	  print_a_vector(zm1,k);
	  }*/
        
	
	//Create local reduced matrix elements on every processor
	vector<double> s_local(2);
	vector<double> r_local(2);
	vector<double> t_local(2);
	vector<double> u_local(2);

	int count =0;
	
	if(my_rank ==0){
	  s_local[0] = (zm[k-1]);
	  t_local[0] = (1/a_const);                                         
	  u_local[0] = (-y[k-1]);
	}
	else if(my_rank == comm_sz-1){
	  if ((2*comm_sz-2)%2 == 0){
	    s_local[0] = (zm[0]);
	    r_local[0] = (1/c_const);                                   
	    u_local[0] = (-y[0]);
	  }
	  else {
	    s_local[0] = (zm[k-1]);
	    u_local[0] = (-y[k-1]);
	  }
	}
	else {

	    s_local[0] = zm[0];
	    s_local[1] = zm1[k-1];
	    r_local[0] = (1/c_const);                                                                            
	    r_local[1] = (zm[k-1]);
	    t_local[0] = (zm1[0]);
	    t_local[1] = (1/a_const);                                                        
	    u_local[0] = (-y[0]);
	    u_local[1] = (-y[k-1]);
	  
	}

	gather_start = MPI::Wtime();

	//Pack into communication packet
	package2.insert(package2.end(),s_local.begin(),s_local.end());                   //
	package2.insert(package2.end(),r_local.begin(),r_local.end());                   //  Combine vectors into one vector before gather 
	package2.insert(package2.end(),t_local.begin(),t_local.end());                   //  to rank 0
	package2.insert(package2.end(),u_local.begin(),u_local.end());                   //

	
	//Gather results back to rank 0
	comm.Gather(&package2.front(),8,MPI_DOUBLE,&recieve.front(),8,MPI_DOUBLE,0);     //  Gather 8 values from each processor back to rank 0


	//Unpack on rank 0
	if (my_rank ==0){                                                                //
	  int count = 0;                                                                 //
	  for (int i =0; i < 8*comm_sz; i+=8){                                           //
	    if (i == 0){                                                                 //
	      s[0] = recieve[0];                                                         //
	      t[0] = recieve[4];                                                         //
	      u[0] = recieve[6];                                                         //
	      count ++;                                                                  //
	    }                                                                            //
	    else if (i == 8*(comm_sz-1)){                                                //
	        s[count] = recieve[i];                                                   //
		r[count-1] = recieve[i+2];                                               //
		u[count] = recieve[i+6];                                                 //
		count++;                                                                 //   Unpack the gathered elements on Rank 0
	    }                                                                            //
	    else{                                                                        //
	      s[count] = recieve[i];                                                     //
	      s[count+1] = recieve[i+1];                                                 //
	      r[count-1] = recieve[i+2];                                                 //
	      r[count] = recieve[i+3];                                                   //
	      t[count] = recieve[i+4];                                                   //
	      t[count+1] = recieve[i+5];                                                 //
	      u[count] = recieve[i+6];                                                   //
	      u[count+1] = recieve[i+7];                                                 //
              count+=2;                                                                  //
	    }                                                                            //
	  }                                                                              //
	}                                                                                //

	gather_finish = MPI::Wtime();

	//Solve new matrix on rank 0
	if(my_rank == 0){	
	
	  u = thomas_algorithm(r,s,t,u,2*comm_sz-2);

	}
	
	//Bcast new matrix solutions
	bcast_start = MPI::Wtime();
	comm.Bcast(&u.front(),2*comm_sz-2,MPI_DOUBLE,0);
	bcast_finish = MPI::Wtime();

       	//Solve for local solution "x" using solution of the reduced matrix
	vector<double> x(k);
	int index = 2*(my_rank+1)-3;
	if (my_rank ==0){
	  index =0;
	  for (int i = 0; i < k; i++){
		x[i] = y[i] + u[index]*zm[i];
	  }
	}
	else if (my_rank == comm_sz-1){
	  for (int i = 0; i < k; i++){
		x[i] = y[i] + u[index]*zm[i];
	  }
	}
	else {
	  for (int i =0; i < k; i++){
	    x[i] = y[i] + u[index]*zm[i] + u[index+1]*zm1[i];
	  }
	}

	
	//Time finish
	finish_time = MPI::Wtime();

	vector<double> x_all(N);

	//Gather all solutions back to rank 0 to print and check values
	comm.Gather(&x.front(), k, MPI_DOUBLE, &x_all.front(),k,MPI_DOUBLE,0);

	//Code to print and check solution
	if(my_rank ==0){
	  
	  double err_sum = 0.0;
	  double value = 0.0;
	  for (int i= 0; i < N; i++){
	    value = x_all[i] - double(i+1);
	    //cout << "x: " << x_all[i] << " i " << double(i+1) << " " << value << " ";
	    value = abs(value);
	    //cout << value << " ";
	    err_sum += value;
	    //cout << "err_sum " << err_sum << endl;
	  }
	  err_sum /= N;


	  //Print Results
	  cout << endl <<  "For problem size N = " << N << " and " << comm_sz << " processors: " << endl;
	  cout << "Average error is: ";
	  cout << err_sum << endl;
	  cout << "Run time is " << finish_time-start_time << endl;
	  cout << "Communication time is: " << (bcast_finish - bcast_start) + (gather_finish-gather_start) << endl;
	  cout << "Bcast time is: " << bcast_finish - bcast_start << endl;

	  vector<double> solutions(N);

	  //Record serial time
	  start_time = MPI::Wtime();
	  solutions = thomas_algorithm(a,b,c,sol,N);
	  finish_time = MPI::Wtime();

	  cout << "Serial time is " << finish_time-start_time << endl;
	}

	
	MPI::Finalize();

}


