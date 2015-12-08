package ece.ncsu.edu.gpucompiler.cuda.cetus;

public class UnsupportedCodeException extends Exception {
    /**
	 * 
	 */
	private static final long serialVersionUID = -8203353680818240228L;


	/**
     * Constructs an UnsupportedOperationException with no detail message.
     */
    public UnsupportedCodeException() {
    }


    public UnsupportedCodeException(String message) {
	super(message);
    }


    public UnsupportedCodeException(String message, Throwable cause) {
        super(message, cause);
    }
 

    public UnsupportedCodeException(Throwable cause) {
        super(cause);
    }
}
