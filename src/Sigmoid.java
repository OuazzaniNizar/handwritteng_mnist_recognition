public class Sigmoid implements Activation{

	/**
	 * The logistic function used as an activation function for the neuron
	 * @param val
	 * @return
	 */
	public double activate(double val){
		return 1.0/(1.0+Math.exp(-val));
	}

}