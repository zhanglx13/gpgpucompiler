package ece.ncsu.edu.gpucompiler.cuda.index;

import java.util.ArrayList;
import java.util.List;

import cetus.analysis.NormalExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.IntegerLiteral;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import ece.ncsu.edu.gpucompiler.cuda.cetus.GLoop;
import ece.ncsu.edu.gpucompiler.cuda.cetus.UnsupportedCodeException;

public class Address {
	protected List<Index> indexs = new ArrayList<Index>();
	
	public Address() {}
	
	public String toString() {
		StringBuffer sb = new StringBuffer("address:");	
		for (Index index: indexs) {
			sb.append(index.toString()).append(";");
		}
		return sb.toString();
	}

	public List<Index> getIndexs() {
		return indexs;
	}

	public void setIndexs(List<Index> indexs) {
		this.indexs = indexs;
	}
	

	public boolean isZero() {
		if (getIndexs().size()==0) {
			return true;
		}
		if (getIndexs().size()==1) {
			Index in = getIndexs().get(0);
			if (in instanceof ConstIndex) {
				if (in.getCoefficient()==0) {
					return true;
				}
			}
		}
		return false;
		
	}
	
	/**
	 * replace index to index+indexb
	 * @param index
	 * @param indexb
	 */
	public void replace(Index index, Index indexb) {
		ArrayList<Index> list = isContain(index);
		for (Index in: list) {
			Index newin = (Index)indexb.clone();
			newin.setCoefficient(in.getCoefficient());
			this.getIndexs().add(newin);
		}
	}
	
	
	public ArrayList<Index> isContain(Index index) {
		ArrayList<Index> list = new ArrayList();
		for (Index in: indexs) {
			if (in.getClass().equals(index.getClass())) {
				if (in instanceof ThreadIndex) {
					ThreadIndex in0 = (ThreadIndex)in;
					ThreadIndex index0 = (ThreadIndex)index;
					if (index0.getId()==null||in0.getId().equals(index0.getId())) 
						list.add(in);
				}
				else 
				if (in instanceof ConstIndex) {
					list.add(in);
				}
				else
				if (in instanceof LoopIndex) {
					LoopIndex index0 = (LoopIndex)index;
					LoopIndex in0 = (LoopIndex)in;
					if (index0.getId()==null||in0.getId().equals(index0.getId()))
						list.add(in);					
				}
			}
		}
		return list;
	}
	
	public ArrayList<Index> isContain(Class clazz) {
		ArrayList<Index> list = new ArrayList();
		for (Index in: indexs) {
			if (in.getClass().equals(clazz)) list.add(in);
		}
		return list;
	}
	
	public int computeOffset(Index index, int offset) {
		for (Index in: indexs) {
			if (in.getClass().equals(index.getClass())) {
				if (in instanceof ThreadIndex) {
					ThreadIndex in0 = (ThreadIndex)in;
					ThreadIndex index0 = (ThreadIndex)index;
					if (index0.getId()==null||in0.getId().equals(index0.getId())) {
						if (in.getCoefficient()<0) return -offset;
						else return offset;						
					}
				}
				else
				if (in instanceof LoopIndex) {
					LoopIndex index0 = (LoopIndex)index;
					LoopIndex in0 = (LoopIndex)in;
					if (index0.getId()==null||in0.getId().equals(index0.getId())) {
						if (in.getCoefficient()<0) return -offset;
						else return offset;												
					}
				}				
				else {
					if (in.getCoefficient()<0) return -offset;
					else return offset;
				}
					
			}
		}
		return 0;
	}
	
	
	static void parse(Expression ex, GLoop loop, Address address, int coe) throws UnsupportedCodeException {
		Index in = null;
		if (ex instanceof Identifier) {
			Identifier id = (Identifier)ex;
			if (loop.getIterators().contains(id)||id.equals(loop.getIterator())) {
				LoopIndex index = (new LoopIndex(id));
				index.setLoop(loop);
				index.setCoefficient(coe);
				in = index;
			}
			else {
				Index index = ThreadIndex.getThreadIndex(id.getName());
				if (index!=null) {
					index.setCoefficient(coe);
					in = index;
				}
				else {
					UnresolvedIndex ui = new UnresolvedIndex(id);
					ui.setCoefficient(coe);
					in = ui;
				}
			}
			
		}
		else if (ex instanceof IntegerLiteral) {
			IntegerLiteral il = (IntegerLiteral)ex;
			ConstIndex constIndex = new ConstIndex(coe*(int)il.getValue());
//			Index index = (constIndex);			
//			index.setCoefficient(coe);
			in = constIndex;
		}		
		else
		if (ex instanceof BinaryExpression) {
			BinaryExpression be = (BinaryExpression)ex;
			Expression lhs = be.getLHS();
			BinaryOperator op = be.getOperator();
			Expression rhs = be.getRHS();
			if (BinaryOperator.SUBTRACT==op) {
				parse(lhs, loop, address, coe);
				parse(rhs, loop, address, -coe);				
			}
			else 
			if (BinaryOperator.ADD==op) {
				parse(lhs, loop, address, coe);
				parse(rhs, loop, address, coe);								
			}
			else
			if (BinaryOperator.MULTIPLY==op) {
				if (lhs instanceof UnaryExpression) {
					UnaryExpression ue = (UnaryExpression)ex;
					if (ue.getOperator().equals(UnaryOperator.MINUS)) {
						coe = -coe;
					}
					lhs = ue.getExpression();
				}
				if (rhs instanceof UnaryExpression) {
					UnaryExpression ue = (UnaryExpression)ex;
					if (ue.getOperator().equals(UnaryOperator.MINUS)) {
						coe = -coe;
					}
					rhs = ue.getExpression();
				}
				
				if (lhs instanceof IntegerLiteral) {
					int v = (int)((IntegerLiteral)lhs).getValue();
					if (v!=0)
						parse(rhs, loop, address, v*coe);								
				}
				else
				if (rhs instanceof IntegerLiteral) {
					int v = (int)((IntegerLiteral)rhs).getValue();
					if (v!=0)
						parse(lhs, loop, address, v*coe);								
				}
				else {
					/**
					 * for final result, we can accept it, but we cannot parse it
					 */
					System.err.println("the address of memory cannot contain a*b, while a and b are all variables:"+ex+":"+lhs.getClass()+":"+rhs);
//					throw new UnsupportedCodeException("the address of memory cannot contain a*b, while a and b are all variables:"+ex+":"+lhs.getClass()+":"+rhs);
					
				}
					
			}
		}
		else 
		if (ex instanceof UnaryExpression) {
			UnaryExpression ue = (UnaryExpression)ex;
			if (ue.getOperator().equals(UnaryOperator.MINUS)) {
				parse(ue.getExpression(), loop, address, -coe);
			}
			else
			if (ue.getOperator().equals(UnaryOperator.PLUS)) {
				parse(ue.getExpression(), loop, address, coe);
			}
			else {
				System.err.println("(Address) cannot handle expression:" + ex);				
			}
		}
		else {
//			throw new Exception("cannot parse expression to address unexpected "+ex);
			System.err.println("(Address) cannot handle expression:" + ex);			
			System.err.println(ex.getClass());			
		}
		if (in!=null) {
//			in.setCoefficient(coe);
//			System.out.println("add:"+in+":"+coe);
			address.getIndexs().add(in);			
		}
		
	}	
	public static Address parseAddress(Expression ex, GLoop loop) throws UnsupportedCodeException {
		if (ex==null) return new Address();
		Address address = new Address();
		ex = NormalExpression.simplify(ex);
//		System.out.println(ex);
		parse(ex, loop, address, 1);
//		System.out.println(address.toExpression());
		return address;		
	}
	

	public static boolean isCoalesced(Address addressx, Address addressy) {
		if (addressy!=null)
		for (Index index: addressy.getIndexs()) {
			if (index instanceof ThreadIndex) {
				// y coordinator cannot include idx
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.IDX)) return false;
			}
		}


		boolean iscolaescd = false;
		for (Index index: addressx.getIndexs()) {
			if (index instanceof ConstIndex) {
				ConstIndex ci = (ConstIndex)index;
				// constant value only can be 16x
				if (ci.getCoefficient()%16!=0) return false;
			}
			else 
			if (index instanceof ThreadIndex) {
				ThreadIndex ti = (ThreadIndex)index;
				if (ti.getId().equals(ThreadIndex.IDX)&&ti.getCoefficient()==1) {
					iscolaescd = true;				
				}
			}
			else
			if (index instanceof LoopIndex) {
//				LoopIndex li = (LoopIndex)index;
				return false;
			}
		}
		return iscolaescd;
	}
	
	public Address subtract(Address addr) {
		Address result = new Address();
		for (Index index: this.getIndexs()) {
			index = (Index)index.clone();
			result.getIndexs().add(index);
//				System.out.println("add:"+index);
		}
		for (Index index: addr.getIndexs()) {
			index = (Index)index.clone();
			index.setCoefficient(-index.getCoefficient());
			result.getIndexs().add(index);
//				System.out.println("add:"+index);
		}
		result.compact();
		return result;
	}
	

	public void compact() {
		ConstIndex ci = null;
		for (int i=0; i<indexs.size(); i++) {
			Index index = indexs.get(i);
//			System.out.println(index);
			if (index instanceof ConstIndex) {
				ConstIndex in = (ConstIndex)index;
				if (ci==null) {
					ci = in;
				}
				else {
					int a = ci.getCoefficient()+in.getCoefficient();
					ci.setCoefficient(a);
				}
				indexs.remove(i);
				i--;
			}
			else
			if (index instanceof ThreadIndex) {
				ThreadIndex in = ((ThreadIndex) index);
//				System.out.println(in+";");
				for (int j=0; j<i; j++) {
					Index index0 = indexs.get(j);
					if (index0 instanceof ThreadIndex) {
						ThreadIndex in0 = ((ThreadIndex) index0);						
//						System.out.println(index0+":"+in0.getId().equals(in.getId())+":"+(in0.isNegative())+"-"+(in.isNegative()));
						if (in0.getId().equals(in.getId())) {
							int coe = in.getCoefficient()+in0.getCoefficient();
//							System.out.println(coe+";"+in+";"+in0);
							if (coe==0) {
								indexs.remove(in);
								i=i-2;
							}
							else {
								in.setCoefficient(coe);
								i=i-1;
							}
							indexs.remove(in0);
							continue;
						}
					}
				}
			}
			else
			if (index instanceof LoopIndex) {
				LoopIndex in =  ((LoopIndex) index);
				for (int j=0; j<i; j++) {
					Index index0 = indexs.get(j);
					if (index0 instanceof LoopIndex) {
						LoopIndex in0 = ((LoopIndex) index0);						
						if (in0.getId().equals(in.getId())) {
							int coe = in.getCoefficient()+in0.getCoefficient();
							if (coe==0) {
								indexs.remove(in);
								i=i-2;
							}
							else {
								in.setCoefficient(coe);
								i=i-1;
							}
							indexs.remove(in0);
							continue;
						}
					}
				}				
			}
			else
			if (index instanceof UnresolvedIndex) {
				UnresolvedIndex in =  ((UnresolvedIndex) index);
				for (int j=0; j<i; j++) {
					Index index0 = indexs.get(j);
					if (index0 instanceof UnresolvedIndex) {
						UnresolvedIndex in0 = ((UnresolvedIndex) index0);						
						if (in0.getId().equals(in.getId())) {
							int coe = in.getCoefficient()+in0.getCoefficient();
							if (coe==0) {
								indexs.remove(in);
								i=i-2;
								indexs.remove(in0);
								continue;
							}
						}
					}
				}	
			}			
		}
		
		if (ci!=null&&ci.getCoefficient()!=0) {
			indexs.add(ci);
		}
	}
	
	public Expression toExpression() {
		Expression previous = null;
		compact();
		
		for (int i=0; i<indexs.size(); i++) {
			Index index = indexs.get(i);
			Expression current = null;
			if (index instanceof ConstIndex) {
				current = new IntegerLiteral(index.getCoefficient());
			}
			else
			if (index instanceof ThreadIndex) {
				current = new Identifier(((ThreadIndex) index).getId());
			}
			else
			if (index instanceof LoopIndex) {
				current =  ((LoopIndex) index).getId();
			}
			else
			if (index instanceof UnresolvedIndex) {
				UnresolvedIndex ui = (UnresolvedIndex)index;
				current = ui.getId();
			}
			
			if (index instanceof ConstIndex) {
			}
			else
			if (index.getCoefficient()!=1) {
				current = new BinaryExpression(new IntegerLiteral(index.getCoefficient()), BinaryOperator.MULTIPLY, current);
			}

			if (previous==null) {
				previous = current;
			}
			else {
				previous = new BinaryExpression(previous, BinaryOperator.ADD, current);
			}
		}
		if (previous==null) {
			previous = new IntegerLiteral(0);
		}
//		previous = NormalExpression.simplify(previous);
		return previous;
	}
	
	public Object clone() {
		Address obj = new Address();
		for (Index index: indexs) {
			obj.indexs.add((Index)index.clone());
		}
		return obj;
	}	
}
