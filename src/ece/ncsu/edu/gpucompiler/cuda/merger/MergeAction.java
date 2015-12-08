package ece.ncsu.edu.gpucompiler.cuda.merger;

import java.util.ArrayList;
import java.util.List;

public class MergeAction {
	int mergeDirection;
	int mergeType;
	int mergeNumber;
	List<BlockDiff> blockDiffs = new ArrayList();
	public int getMergeDirection() {
		return mergeDirection;
	}
	public void setMergeDirection(int mergeDirection) {
		this.mergeDirection = mergeDirection;
	}
	public int getMergeType() {
		return mergeType;
	}
	public void setMergeType(int mergeType) {
		this.mergeType = mergeType;
	}
	public int getMergeNumber() {
		return mergeNumber;
	}
	public void setMergeNumber(int mergeNumber) {
		this.mergeNumber = mergeNumber;
	}

	public List<BlockDiff> getBlockDiffs() {
		return blockDiffs;
	}
	public void setBlockDiffs(List<BlockDiff> blockDiffs) {
		this.blockDiffs = blockDiffs;
	}
	@Override
	public String toString() {
		
		return (mergeType==Merger.MERGE_TYPE_THREAD?"THREAD":"THREADBLOCK")+";"+
		(mergeDirection==Merger.MERGE_DIRECTION_X?"X":"Y")+";"+
		""+mergeNumber+";";
		
	}
	
	
}