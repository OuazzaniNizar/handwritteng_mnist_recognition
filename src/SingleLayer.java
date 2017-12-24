import java.util.*;

public class SingleLayer extends ANN{
	
	
	

	public SingleLayer(Map<Input, Output> trainingData, Map<Input, Output> testingData) {
		generator = new Random();
		this.trainingData = trainingData;
		this.testingData = testingData;

		// to be completed
		this.inLayer= new ArrayList<>();
		for(int i=0;i<28*28;i++){
			inLayer.add(new InputNeuron(28*28));
			inLayer.get(i).name="inputNeuron"+i;
			inLayer.get(i).feed(0);
		}

		this.outLayer= new ArrayList<>();
		for(int i=0;i<10;i++){
			outLayer.add(new Neuron(new Sigmoid()));
			outLayer.get(i).name="outputNeuron"+i;
			outLayer.get(i).out=0;
		}
		for(int neuronSorite=0;neuronSorite<10;neuronSorite++){
			for(int j=0;j<inLayer.size();j++){
				outLayer.get(neuronSorite).addParent(inLayer.get(j));
			}
			outLayer.get(neuronSorite).initWeights();
		}

	}


	/**
	 * Methods that computes the output of the neural network given the input
	 * @param in input data of the neural network
	 * @return the output of the neural network (it is stored in an object output that contains a vector
	 *  of the values of each neurons in the output layer
	 */
	public Output feed(Input in){
		// to be completed
		initialiseInputLayer(in);
		double[] output = new double[10];
		for (int i=0;i<10;i++){
			outLayer.get(i).feed();
			output[i]=outLayer.get(i).out;
		}
		Output out= new Output(output);
		return out;
	}

	//Pour une Input on initialise l'Input layer avec les pixels de l'image
	public void initialiseInputLayer(Input in){
		int numInput=0;
		for(int ligne=0;ligne<in.data.length;ligne++){
			for (int colonne=0;colonne<in.data.length;colonne++){
				inLayer.get(numInput).feed(in.data[ligne][colonne]);
				numInput++;
			}
		}
	}


	/**
	 * method that trains the neural network for a certain number of iterations (no convergence test is used).
	 * @param nbIterations is the number of iterations, i.e. the number of times the algorithm will update using
	 *  the entire training data
	 * @return returns the dynamics of the error: it contains a map that associate one iteration number and the
	 * number of mistakes done on the testing set.
	 *
	 */
	public Map<Integer,Double> train(int nbIterations) {
		// to be completed
		Map<Integer,Double> data=new HashMap<Integer, Double>();
		long startTime = System.currentTimeMillis();


		for(int iter=0; iter<nbIterations;iter++){
			int traitement=0;
				for (Map.Entry<Input, Output> d : trainingData.entrySet()) {
					traitement++;
					Input in = d.getKey();
					Output realOutput = d.getValue();
					//Feedforwad
					feed(in);
					//Feedbackward
					for(int sortie=0; sortie<10;sortie++){
						outLayer.get(sortie).backPropagate(realOutput.value[sortie]);
					}
					if(traitement%100==0)	System.out.println("["+iter+"] - Traitement de l'image "+traitement+" terminé ");
				}
			//A chaque itération i on calcule l'erreur globale dans cette itération SUR LES DONNEES DE TEST
			data.put(iter,test(testingData,iter));
			System.out.println("SingleLayer-------------------FIN ITERATION N°"+iter+"-------------------");
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Temps d'execution :" + (endTime - startTime)/(1000*60) + " minutes");//différence donnée en millisecondes..
		return data;
	}
	

}