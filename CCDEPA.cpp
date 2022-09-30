#include <iostream>
#include <mpi/mpi.h>
#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <string>
#include <float.h>
#include <vector>
#include <set>
#include <iterator>
#include "Header.h"
#include <ctime>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>

using namespace std;

#define SIZEACHT 30
#define INITALLP 2
#define MAXTEACHP 4
#define MINTEACHP 1
//#define INIGEACH 10
#define DIM 1000
#define GENERATI 500

struct NSFC
{
	int ns1;
	int ns2;
	int nf1;
	int nf2;
	double wcr;
	double deltaf;
	int best;
	NSFC() {
        ns1 = ns2 = nf1 = nf2 = wcr = deltaf = 0;
	}
};

unsigned seed = time(NULL);
default_random_engine gen(seed);
uniform_real_distribution<double> dis(0.0, 1.0);
normal_distribution<double> ndis(0.5, 0.3);
cauchy_distribution<double> cdis(0, 1);

double* swarm[SIZEACHT * MAXTEACHP];
double* swarm2[SIZEACHT * MAXTEACHP];
double fitne[SIZEACHT * MAXTEACHP];
int D;
int* dims;
double gbest[DIM];
Benchmarks* fp[MAXTEACHP];
int threadNum;
double crv[SIZEACHT * MAXTEACHP];
double crm = 0.5;
double pf = 0.5;
int curBest = 0;

Benchmarks* generateFuncObj(int funcID){
    Benchmarks *fp;
    // run each of specified function in "configure.ini"
    if (funcID==1){
        fp = new F1();
    }else if (funcID==2){
        fp = new F2();
    }else if (funcID==3){
        fp = new F3();
    }else if (funcID==4){
        fp = new F4();
    }else if (funcID==5){
        fp = new F5();
    }else if (funcID==6){
        fp = new F6();
    }else if (funcID==7){
        fp = new F7();
    }else if (funcID==8){
        fp = new F8();
    }else if (funcID==9){
        fp = new F9();
    }else if (funcID==10){
        fp = new F10();
    }else if (funcID==11){
        fp = new F11();
    }else if (funcID==12){
        fp = new F12();
    }else if (funcID==13){
        fp = new F13();
    }else if (funcID==14){
        fp = new F14();
    }else if (funcID==15){
        fp = new F15();
    }else{
        cerr<<"Fail to locate Specified Function Index"<<endl;
        exit(-1);
    }
    return fp;
}

void* initialization(void* startP) {
	int startPoint = *((int*)startP);
	int theFp = startPoint / SIZEACHT;
	for (int i = startPoint; i < startPoint + SIZEACHT; i++) {
		for (int j = 0; j < D; j++) {
			swarm2[i][j] = swarm[i][j] = dis(gen) * (fp[theFp]->getMaxX() - fp[theFp]->getMinX()) + fp[theFp]->getMinX();
		}
	}
	double trial[DIM];
	memcpy(trial, gbest, sizeof(double) * DIM);
	int* theBest = new int;
	*theBest = startPoint;
	for (int i = startPoint; i < startPoint + SIZEACHT; i++) {
		for (int j = 0; j < D; j++) {
			trial[dims[j]] = swarm[i][j];
		}
		fitne[i] = fp[theFp]->compute(trial);
		if (fitne[*theBest] > fitne[i]) *theBest = i;
	}
	return theBest;
}

void* recalculate(void* startP) {
    int startPoint = *((int*)startP);
    int theFp = startPoint / SIZEACHT;
    double trial[DIM];
    memcpy(trial, gbest, sizeof(double) * DIM);
    int* theBest = new int;
    *theBest = startPoint;
    for (int i = startPoint; i < startPoint + SIZEACHT; i++) {
        for (int j = 0; j < D; j++) {
            trial[dims[j]] = swarm[i][j];
        }
        fitne[i] = fp[theFp]->compute(trial);
        if (fitne[*theBest] > fitne[i]) *theBest = i;
    }
    return (void *)theBest;
}

void* funcMain(void* startP) {
	int startPoint = *((int*)startP);
	int theFp = startPoint / SIZEACHT;
	double temp[DIM];
	memcpy(temp, gbest, sizeof(double) * DIM);
    double* trial = new double[D];
    NSFC* nsfcp = new NSFC();
    nsfcp->best = startPoint;
    for (int i = startPoint; i < startPoint + SIZEACHT; i++) {
        if (dis(gen) < pf) {
            int a, b, c;
            do
            {
                a = dis(gen) * SIZEACHT * threadNum;
            } while (a == i);
            do
            {
                b = dis(gen) * SIZEACHT * threadNum;
            } while (a == b || b == i);
            do
            {
                c = dis(gen) * SIZEACHT * threadNum;
            } while (c == a || c == b || c == i);
            int j = dis(gen) * D;
            bool flag = true;
            double F = 0;
            if (dis(gen) < pf) {
                F = ndis(gen);
            }
            else {
                F = cdis(gen);
            }
            for (int k = 1; k <= D; k++) {
                if (dis(gen) < crv[i] || k == D) {
                    trial[j] = swarm[c][j] + F * (swarm[a][j] - swarm[b][j]);
                    if (trial[j] > fp[theFp]->getMaxX() || trial[j] < fp[theFp]->getMinX())
                        flag = false;
                }
                else trial[j] = swarm[i][j];
                j = (j + 1) % D;
            }
            double score = DBL_MAX;
            if (flag) {
                for (int j = 0; j < D; j++) {
                    temp[dims[j]] = trial[j];
                }
                score = fp[theFp]->compute(temp);
            }
            if (score <= fitne[i]) {
                memcpy(swarm2[i], trial, sizeof(double) * D);
                (nsfcp->ns1)++;
                double de = fitne[i] - score;
                nsfcp->deltaf += de;
                nsfcp->wcr += (de * crv[i]);
                fitne[i] = score;
            }
            else {
                memcpy(swarm2[i], swarm[i], sizeof(double) * D);
                nsfcp->nf1++;
            }
            if (fitne[nsfcp->best] > fitne[i]) nsfcp->best = i;
        }
        else {
            int a, b;
            do
            {
                a = dis(gen) * SIZEACHT * threadNum;
            } while (a == i);
            do
            {
                b = dis(gen) * SIZEACHT * threadNum;
            } while (a == b || b == i);
            int j = dis(gen) * D;
            bool flag = true;
            double F = 0;
            if (dis(gen) < pf) {
                F = ndis(gen);
            }
            else {
                F = cdis(gen);
            }
            for (int k = 1; k <= D; k++) {
                if (dis(gen) < crv[i] || k == D) {
                    trial[j] = swarm[i][j] + F * (swarm[a][j] - swarm[b][j]) + F * (swarm[curBest][j] - swarm[i][j]);
                    if (trial[j] > fp[theFp]->getMaxX() || trial[j] < fp[theFp]->getMinX())
                        flag = false;
                }
                else trial[j] = swarm[i][j];
                j = (j + 1) % D;
            }
            double score = DBL_MAX;
            if (flag) {
                for (int j = 0; j < D; j++) {
                    temp[dims[j]] = trial[j];
                }
                score = fp[theFp]->compute(temp);
            }
            if (score <= fitne[i]) {
                memcpy(swarm2[i], trial, sizeof(double) * D);
                (nsfcp->ns2)++;
                double de = fitne[i] - score;
                nsfcp->deltaf += de;
                nsfcp->wcr += (de * crv[i]);
                fitne[i] = score;
            }
            else {
                memcpy(swarm2[i], swarm[i], sizeof(double) * D);
                nsfcp->nf2++;
            }
            if (fitne[nsfcp->best] > fitne[i]) nsfcp->best = i;
        }
    }
    delete[] trial;
    return nsfcp;
}

int main(int argc, char* argv[]) {
    int id, totalp;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &totalp);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Status status;
    int theFunc = atoi(argv[1]);
    for (int i = 0; i < MAXTEACHP; i++) {
        fp[i] = generateFuncObj(theFunc);
    }
    ofstream result;
    ifstream groupFile;
    if (id == 0) {
        stringstream ss;
        ss << theFunc;
        string funStr;
        ss >> funStr;
        string fileName = funStr + "_result_P.txt";
        result.open(fileName.c_str(), ios::app);
        fileName = funStr + "_groupFile.txt";
        groupFile.open(fileName.c_str());
    }
    vector< vector<int> > groups;
    if (id == 0) {
        int groupNum;
        groupFile >> groupNum;
        for (int i = 0; i < groupNum; i++) {
            int eleNum;
            std::vector<int> v;
            groupFile >> eleNum;
            for (int j = 0; j < eleNum; j++) {
                int ele;
                groupFile >> ele;
                v.push_back(ele);
            }
            groups.push_back(v);
        }
        for (int i = 1; i < totalp; i++) {
            int* dd = groups[i].data();
            int leng = groups[i].size();
            MPI_Send(&leng, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(dd, leng, MPI_INT, i, 2, MPI_COMM_WORLD);
        }
        D = groups[0].size();
        dims = new int[D];
        for (int i = 0; i < D; i++) {
            dims[i] = groups[0][i];
        }
        groupFile.close();
    }
    else {
        MPI_Recv(&D, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        dims = new int[D];
        MPI_Recv(dims, D, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    }
    for (int i = 0; i < SIZEACHT * MAXTEACHP; i++) {
        swarm2[i] = new double[D];
        swarm[i] = new double[D];
    }
    //gbest generation
    if (id == 0) {
    	for (int i = 0; i < DIM; i++) {
    		gbest[i] = fp[0]->getMinX() + (fp[0]->getMaxX() - fp[0]->getMinX()) * dis(gen);
    	}
    }
    MPI_Bcast(gbest, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //resource allocation generation
    int* tnEp;
    if (id == 0) {
    	tnEp = new int[totalp];
    	for (int i = 0; i < totalp; i++) {
    		tnEp[i] = INITALLP;
    	}
    }
    int threadNum2;
    MPI_Scatter(tnEp, 1, MPI_INT, &threadNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //initialization
    pthread_t child[MAXTEACHP];
    int startPoint[MAXTEACHP];
    for (int i = 0; i < MAXTEACHP; i++) {
        startPoint[i] = i * SIZEACHT;
    }
    for (int i = 1; i < threadNum; i++) {
        pthread_create(&child[i], NULL, initialization, &startPoint[i]);
    }
    void* fir = initialization(&startPoint[0]);
    curBest = *((int*)fir);
    delete (int*)fir;
    for (int i = 1; i < threadNum; i++) {
    	void* sta;
    	pthread_join(child[i], &sta);
    	int x = *((int*)sta);
    	if (fitne[curBest] > fitne[x]) curBest = x;
    	delete (int*)sta;
    }
    if (id == 0) {
    	double forGbestMerge[DIM];
    	for (int i = 1; i < totalp; i++) {
    		MPI_Recv(forGbestMerge, groups[i].size(), MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);
    		for (int j = 0; j < groups[i].size(); j++) {
    			gbest[groups[i][j]] = forGbestMerge[j];
    		}
    	}
    	for (int i = 0; i < groups[0].size(); i++) {
    		gbest[groups[0][i]] = swarm[curBest][i];
    	}
    }
    else {
    	MPI_Send(swarm[curBest], D, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
    MPI_Bcast(gbest, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double difference = 0;
    double* diffeFit;
    if (id == 0) {
        diffeFit = new double [totalp];
    }
    int counter = 0;
    double ns1 = 0, ns2 = 0, nf1 = 0, nf2 = 0, wcr = 0, deltaf = 0;
    normal_distribution<double> crmdis(crm, 0.1);
    for (int i = 0; i < SIZEACHT * threadNum; i++) {
    	crv[i] = crmdis(gen);
    }
    for (int i = 1; i < GENERATI; i++) {
        //int geNo = threadNum * INIGEACH;
        //for (int j = 0; j < threadNum; j++) {
        //    startPoint[j] = SIZEACHT * j;
        //}
        //recalculate
        for (int j = 1; j < threadNum; j++) {
            pthread_create(&child[j], NULL, recalculate, &startPoint[j]);
        }
        fir = recalculate(&startPoint[0]);
        curBest = *((int*)fir);
        delete (int*)fir;
        for (int j = 1; j < threadNum; j++) {
            void* sta;
            pthread_join(child[j], &sta);
            int x = *((int*)sta);
            if (fitne[curBest] > fitne[x]) curBest = x;
            delete (int*)sta;
        }
        difference = fitne[curBest];
        //main loop and func
        for (int j = 0; j < 20; j++) {
            counter++;
            if (counter % 50 == 0) {
                pf = ns1 * (ns2 + nf2) / (ns2 * (ns1 + nf1) + ns1 * (ns2 + nf2));
                ns1 = ns2 = nf1 = nf2 = 0;
            }
            if (counter % 25 == 0) {
                crm = wcr / deltaf;
                wcr = deltaf = 0;
            }
            if (counter % 5 == 0) {
                normal_distribution<double> crmdis2(crm, 0.1);
                for (int k = 0; k < SIZEACHT * threadNum; k++) {
                    crv[k] = crmdis2(gen);
                }
            }
            for (int k = 1; k < threadNum; k++) {
                pthread_create(&child[k], NULL, funcMain, &startPoint[k]);
            }
            fir = funcMain(&startPoint[0]);
            NSFC* nsfcp = (NSFC*)fir;
            ns1 += nsfcp->ns1;
            ns2 += nsfcp->ns2;
            nf1 += nsfcp->nf1;
            nf2 += nsfcp->nf2;
            wcr += nsfcp->wcr;
            deltaf += nsfcp->deltaf;
            curBest = nsfcp->best;
            delete nsfcp;
            for (int k = 1; k < threadNum; k++) {
                void* sta;
                pthread_join(child[k], &sta);
                NSFC* x = (NSFC*)sta;
                if (fitne[curBest] > fitne[x->best]) curBest = x->best;
                ns1 += x->ns1;
                ns2 += x->ns2;
                nf1 += x->nf1;
                nf2 += x->nf2;
                wcr += x->wcr;
                deltaf += x->deltaf;
                delete x;
            }
            for (int k = 0; k < SIZEACHT * threadNum; k++) {
                memcpy(swarm[k], swarm2[k], sizeof(double) * D);
            }
        }
        difference = difference - fitne[curBest];
        if (id == 0) {
            double forGbestMerge[DIM];
            for (int j = 1; j < totalp; j++) {
                MPI_Recv(forGbestMerge, (int)groups[j].size(), MPI_DOUBLE, j, i + 4, MPI_COMM_WORLD, &status);
                for (int k = 0; k < groups[j].size(); k++) {
                    gbest[groups[j][k]] = forGbestMerge[k];
                }
            }
            for (int j = 0; j < D; j++) {
                gbest[groups[0][j]] = swarm[curBest][j];
            }
            double gbestf = fp[0]->compute(gbest);
            result << gbestf << endl;
        }
        else {
            MPI_Send(swarm[curBest], D, MPI_DOUBLE, 0, i + 4, MPI_COMM_WORLD);
        }
        MPI_Bcast(gbest, DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&difference, 1, MPI_DOUBLE, diffeFit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        difference = 0;
    	if (id == 0) {
            std::vector<int> seq;
            for (int j = 0; j < totalp; j++) {
                seq.push_back(j);
            }
            for (int j = 0; j < totalp - 1; j++) {
                for (int k = 0; k < totalp - 1 - j; k++) {
                    if ((diffeFit[seq[k]] / (double)tnEp[seq[k]]) > (diffeFit[seq[k + 1]] / (double)tnEp[seq[k + 1]])) {
                        int temseq = seq[k];
                        seq[k] = seq[k + 1];
                        seq[k + 1] = temseq;
                    }
                }
            }
            int thebig = -1;
            for (int j = totalp - 1; j >= 0; j--) {
                if (tnEp[seq[j]] != MAXTEACHP) {
                    thebig = j;
                    break;
                }
            }
            int thesmall = -1;
            for (int j = 0; j < thebig; j++) {
                if (tnEp[seq[j]] != MINTEACHP) {
                    thesmall = j;
                    break;
                }
            }
            if (thebig != -1 && thesmall != -1) {
                tnEp[seq[thebig]]++;
                tnEp[seq[thesmall]]--;
            }
        }
        MPI_Scatter(tnEp, 1, MPI_INT, &threadNum2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (threadNum > threadNum2) {
            double* xtemp = new double[D];
            for (int j = 0; j < (threadNum - threadNum2) * SIZEACHT; j++) {
                for (int k = 0; k < threadNum * SIZEACHT - j - 1; k++) {
                    if (fitne[k] > fitne[k + 1]) {
                        memcpy(xtemp, swarm[k], sizeof(double) * D);
                        memcpy(swarm[k], swarm[k + 1], sizeof(double) * D);
                        memcpy(swarm[k + 1], xtemp, sizeof(double) * D);
                        double temp = fitne[k];
                        fitne[k] = fitne[k + 1];
                        fitne[k + 1] = temp;
                    }
                }
            }
            delete[] xtemp;
        }
        if (threadNum < threadNum2) {
            double* minx = new double[D];
            double* maxx = new double[D];
            for (int j = 0; j < D; j++) {
                minx[j] = swarm[0][j];
                maxx[j] = swarm[0][j];
                for (int k = 0; k < threadNum * SIZEACHT; k++) {
                    if (minx[j] > swarm[k][j]) minx[j] = swarm[k][j];
                    if (maxx[j] < swarm[k][j]) maxx[j] = swarm[k][j];
                }
            }
            for (int j = threadNum * SIZEACHT; j < threadNum2 * SIZEACHT; j++) {
                for (int k = 0; k < D; k++) {
                    swarm[j][k] = dis(gen) * (maxx[k] - minx[k]) + minx[k];
                }
            }
            delete[] minx;
            delete[] maxx;
        }
        threadNum = threadNum2;
    }
    delete dims;
    for (int i = 0; i < SIZEACHT * MAXTEACHP; i++) {
    	delete[] swarm[i];
    	delete[] swarm2[i];
    }
    if (id == 0) {
    	delete[] tnEp;
    	delete[] diffeFit;
    	result.close();
    }
    MPI_Finalize();
    return 0;
}
