public class Tanh implements Activation{

	/**
	 * The tanh function used as an activation function for neurons
	 * @param v
	 * @return
	 */
	public double activate(double v){
		return Math.tanh(v);
	}
}