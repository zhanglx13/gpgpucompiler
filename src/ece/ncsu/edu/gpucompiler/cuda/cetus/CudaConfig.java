package ece.ncsu.edu.gpucompiler.cuda.cetus;

public class CudaConfig {
	/*
	Threads / Warp
	Warps / Multiprocessor
	Threads / Multiprocessor
	Thread Blocks / Multiprocessor
	Total # of 32-bit registers / Multiprocessor
	Register allocation unit size
	Shared Memory / Multiprocessor (bytes)
	*/
	
	public final static CudaConfig CUDA_1_0 = new CudaConfig("cuda1_0", 32, 24, 768, 8, 8192, 256, 16384);
	public final static CudaConfig CUDA_1_1 = new CudaConfig("cuda1_1", 32, 24, 768, 8, 8192, 256, 16384);
	public final static CudaConfig CUDA_1_2 = new CudaConfig("cuda1_2", 32, 32, 1024, 8, 16384, 512, 16384);
	public final static CudaConfig CUDA_1_3 = new CudaConfig("cuda1_3", 32, 32, 1024, 8, 16384, 512, 16384);
	
	static CudaConfig defaultCudaConfig;
	int threadInWap;
	int warpInMP;
	int threadInMP;
	int threadBlockInMP;
	int registerInMP;
	int threadInBlock;
	int shareMemoryInMP;
	static int coalescedThread = 16;
	String name;
	
	
	
	public CudaConfig(String name, int threadInWap, int warpInMP,int threadInMP,int threadBlockInMP,
			int registerInMP,int threadInBlock, int shareMemoryInMP) {
		this.name = name;
		this.threadInWap = threadInWap;
		this.warpInMP = warpInMP;
		this.threadInMP = threadInMP;
		this.threadBlockInMP = threadBlockInMP;
		this.registerInMP = registerInMP;
		this.threadInBlock = threadInBlock;
		this.shareMemoryInMP = shareMemoryInMP;
	}
	
	public static CudaConfig get(String name) {
		if (CUDA_1_0.name.equals(name)) return CUDA_1_0;
		if (CUDA_1_1.name.equals(name)) return CUDA_1_1;
		if (CUDA_1_2.name.equals(name)) return CUDA_1_2;
		if (CUDA_1_3.name.equals(name)) return CUDA_1_3;
		return null;
	}
	
	public static CudaConfig getDefault() {
		if (defaultCudaConfig!=null) return defaultCudaConfig;
		return CUDA_1_3;
	}
	
	public static void setDefault(CudaConfig cfg) {
		defaultCudaConfig = cfg;
	}

	public int getThreadInWap() {
		return threadInWap;
	}

	public void setThreadInWap(int threadInWap) {
		this.threadInWap = threadInWap;
	}

	public int getWarpInMP() {
		return warpInMP;
	}

	public void setWarpInMP(int warpInMP) {
		this.warpInMP = warpInMP;
	}

	public int getThreadInMP() {
		return threadInMP;
	}

	public void setThreadInMP(int threadInMP) {
		this.threadInMP = threadInMP;
	}

	public int getThreadBlockInMP() {
		return threadBlockInMP;
	}

	public void setThreadBlockInMP(int threadBlockInMP) {
		this.threadBlockInMP = threadBlockInMP;
	}

	public int getRegisterInMP() {
		return registerInMP;
	}

	public void setRegisterInMP(int registerInMP) {
		this.registerInMP = registerInMP;
	}

	public int getThreadInBlock() {
		return threadInBlock;
	}

	public void setThreadInBlock(int registerInThread) {
		this.threadInBlock = registerInThread;
	}

	public int getShareMemoryInMP() {
		return shareMemoryInMP;
	}

	public void setShareMemoryInMP(int shareMemoryInMP) {
		this.shareMemoryInMP = shareMemoryInMP;
	}

	public static int getCoalescedThread() {
		return coalescedThread;
	}

	public static void setCoalescedThread(int coalescedThread) {
		CudaConfig.coalescedThread = coalescedThread;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}


	
	
}
