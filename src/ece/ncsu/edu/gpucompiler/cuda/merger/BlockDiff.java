package ece.ncsu.edu.gpucompiler.cuda.merger;

import ece.ncsu.edu.gpucompiler.cuda.cetus.MemoryExpression;

public class BlockDiff {
	int x = -1;
	int y = -1;
	MemoryExpression memoryExpression;
	MemoryExpression memoryExpressionR;
	
	public int getX() {
		return x;
	}
	public void setX(int x) {
		this.x = x;
	}
	public int getY() {
		return y;
	}
	public void setY(int y) {
		this.y = y;
	}
	public MemoryExpression getMemoryExpression() {
		return memoryExpression;
	}
	public void setMemoryExpression(MemoryExpression memoryExpression) {
		this.memoryExpression = memoryExpression;
	}
	public MemoryExpression getMemoryExpressionR() {
		return memoryExpressionR;
	}
	public void setMemoryExpressionR(MemoryExpression memoryExpressionR) {
		this.memoryExpressionR = memoryExpressionR;
	}
	
	
}
