package cetus.analysis;
import cetus.hir.*;
import java.util.*;

/**
 * Wrapper framework for executing specific data-dependence test on array subscripts
 *
 */
public class DDTest {
	LinkedList<Loop> loopnest;
	HashMap<Loop, LoopInfo> loopinfo;

	public DDTest (LinkedList<Loop> nest, HashMap<Loop, LoopInfo> loopInfo)
	{
			  this.loopnest = nest;
			  this.loopinfo = loopInfo;
	}

	/*
	 * Accepts two access pairs, partitions their subscripts, performs dependence testing
	 * and constructs a set of dependence vectors if dependence exists
	 * @param access1
	 * @param access2
	 * @param DVset
	 * @return 	0 -> NO Dependence
	 * 			1 -> Dependence exists
	 * 			2 -> Access pair contains non-affine subscripts, stop testing the loop! (Ineligible)
	 */
	public int testAccessPair (ArrayAccess access1, ArrayAccess access2, ArrayList<DependenceVector> DVset)
	{
		ArrayList<SubscriptPair> subscriptPairs;
		ArrayList<HashSet<SubscriptPair>> partitions;
		
		/* test dependency only if two array accesses with the same array symbol id and 
		 * have the same dimension 
		 */
		if (access1.getNumIndices() == access2.getNumIndices())
		{
			/* Obtain single subscript pairs and while doing so, check if the subscripts are affine */
			int dimensions = access1.getNumIndices();
			subscriptPairs = new ArrayList<SubscriptPair>(dimensions);
			for (int dim=0; dim < dimensions; dim++)
			{
				SubscriptPair pair = new SubscriptPair(access1.getIndex(dim), 
														access2.getIndex(dim), 
														this.loopnest, 
														this.loopinfo);
				/* IMPORTANT: If pair is not affine, don't proceed with testing !! */
				if (!(pair.is_affine))
					return 2;
				
				subscriptPairs.add(dim, pair);
			}
			
			/* Partition the subscript pairs - currently ignore effects of coupled subscripts */
			partitions = getSubscriptPartitions (subscriptPairs);
			for (HashSet<SubscriptPair> partition: partitions)
			{	
				if (partition.size() == 1) //only singletons -> currently it is always one (smin)
				{
					boolean depExists = testSeparableSubscripts (partition, DVset);
					if (!depExists) return 0;
				}
				else 
				{
					Tools.println("testAccessPair: partition.size()="+partition.size(), 0);
					System.exit(0);
				}
			}
		}
		else
			// Dependence cannot exist between two array accesses with different Symbols or with
			// different number of indices
			return 0;
		
		// Dependence exists
		return 1;
	}
	
	private void mergeVectorSets(ArrayList<DependenceVector> DVset, ArrayList<DependenceVector> DV)
	{
		
		if (DVset.size() > 0)
		{
			ArrayList<DependenceVector> auxDVset = new ArrayList<DependenceVector>();
			auxDVset.addAll(DVset);
			DVset.removeAll(auxDVset);
			if (DVset.size() > 0 )
				System.err.println("Unexpected behavior");
			for (DependenceVector dv : auxDVset)
			{
				for (DependenceVector dv_aug: DV)
				{
					DependenceVector new_dv = new DependenceVector (dv);
					new_dv.mergeWith(dv_aug);
					DVset.add(new_dv);
				}	
			}
		}
		else
		{
			DVset.addAll(DV);
		}
		//for each vector in DVset
		//  replicate it ||DV|| times 
		//  merge each replica with one vector in DV
		//  if vector is not valid, erase it.
		return;
	}
	
	private boolean testSeparableSubscripts(HashSet<SubscriptPair> partition, ArrayList<DependenceVector> DVset)
	{
		boolean depExists;
		//iterate over partitions and get singletons
		ArrayList<DependenceVector> DV = new ArrayList<DependenceVector>();
		SubscriptPair pair = partition.iterator().next(); //get the first (AND ONLY) element
		
/*
		switch (pair.getComplexity())
		{
			case 0:
				Tools.println("** calling testZIV", 2);
				depExists = testZIV(pair, DV);
				break;
				// //case 1:
				//	 //depExists = testSIV(pair);
				//	 //break;
			default:
				Tools.println ("** calling testMIV: Complexity=" + pair.getComplexity(), 2);
				depExists = testMIV(pair, DV);
				if (!depExists)
					return depExists;
				else
				{
					Tools.println("** calling merge routine", 2);
					this.mergeVectorSets(DVset, DV);
					return true;
				}
		}
*/
		Tools.println ("** calling testMIV: Complexity=" + pair.getComplexity(), 2);
		depExists = testMIV(pair, DV);
		if (!depExists)
			return depExists;
		else
		{
			Tools.println("** calling merge routine", 2);
			this.mergeVectorSets(DVset, DV);
			return true;
		}
	}

	// Caution: call this only after all subscriptPairs are found
	private ArrayList<HashSet<SubscriptPair>> getSubscriptPartitions (ArrayList<SubscriptPair> subscriptPairs)
	{
		//for now they are all separable
		ArrayList<HashSet<SubscriptPair>> partitions = new ArrayList<HashSet<SubscriptPair>>();	
		//this may look redundant now, but all the partitions are singletons 
		//containing a SubscriptPair, in the future, a more elaborate 
		//partition algorithm will be incorporated along with a coupled subscript test
		Tools.println("getSubscriptPartitions: subscriptPairs.size()="+subscriptPairs.size(), 2);
		for (SubscriptPair pair : subscriptPairs)
		{
			HashSet<SubscriptPair> new_partition = new HashSet<SubscriptPair>();
			new_partition.add(pair);
			partitions.add(new_partition);
		}	
		return partitions;
	}
	
	private boolean testMIV (SubscriptPair pair, ArrayList<DependenceVector>dependence_vectors)
	{
		ArrayList<DependenceVector> new_dv;
		BanerjeeTest bt = new BanerjeeTest (pair.getSubscript1(), 
											pair.getSubscript2(), 
											pair.getEnclosingLoopsList(), 
											pair.getEnclosingLoopsInfo());
		
		Tools.println("== Calling testAllDependenceVectors", 2);
		new_dv = bt.testAllDependenceVectors(pair.getEnclosingLoopsList());
		if (new_dv.size() == 0)
		{
			return false;
		}
		else
		{	
			dependence_vectors.addAll(new_dv);
			return true;
		}
	}

	private boolean testZIV (SubscriptPair pair)
	{
		Tools.println("Subscript1 = " + pair.getSubscript1().toString(), 2);
		Tools.println("Subscript2 = " + pair.getSubscript2().toString(), 2);
		if (pair.getSubscript1().toString().equals(pair.getSubscript2().toString()))
			return true;
		else
			return false;
	}
}
