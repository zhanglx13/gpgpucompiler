package cetus.analysis;

import java.util.*;

import cetus.hir.*;


/**
 * Class Section represents a list of array subscripts that expresses a subset
 * of the whole array elements. Each element in the list should be an object
 * of {@link Section.ELEMENT}.
 *
 * @see Section.ELEMENT
 * @see Section.MAP
 */
public class Section extends ArrayList<Section.ELEMENT> implements Cloneable
{
	// Dimension
	private int dimension;

	/**
	 * Constructs a section with the specified dimension.
	 *
	 * @param dimension the dimension of the array. -1 for scalar variables.
	 */
	public Section(int dimension)
	{
		super();
		this.dimension = dimension;
	}

	/**
	 * Constructs a section with the specified array access.
	 *
	 * @param acc the array access expression.
	 */
	public Section(ArrayAccess acc)
	{
		this(acc.getNumIndices());

		add(new ELEMENT(acc));
	}


	/**
	 * Clones a section object.
	 *
	 * @return the cloned section.
	 */
	public Object clone()
	{
		Section o = new Section(dimension);

		// Make a deep copy since ArrayList makes only a shallow copy.
		for ( ELEMENT section : this )
			o.add((ELEMENT)section.clone());

		return o;
	}


	/**
	 * Adds a new element in the section.
	 *
	 * @param section the new element to be added.
	 * @return true (as per the general contract of Collection.add).
	 */
	public boolean add(ELEMENT section)
	{
		if ( !contains(section) )
			super.add(section);

		return true;
	}


	/**
	 * Checks if the section is for a scalar variable.
	 */
	public boolean isScalar()
	{
		return ( isEmpty() && dimension == -1 );
	}


	/**
	 * Checks if the section is for an array variable.
	 */
	public boolean isArray()
	{
		return ( dimension > 0 );
	}


	/**
	 * Checks if the section contains the specified variables.
	 */
	public boolean containsSymbols(Set<Symbol> vars)
	{
		for ( ELEMENT section : this )
			for ( Expression e : section )
				if ( Tools.containsSymbols(e, vars) )
					return true;
		return false;
	}


	/**
	 * Expand every section under the constraints given by the range domain.
	 *
	 * @param rd the given range domain.
	 * @param vars the set of symbols to be expanded.
	 */
	public void expandMay(RangeDomain rd, Set<Symbol> vars)
	{
		for ( ELEMENT section : this )
			for ( int i=0; i<dimension; ++i )
			{
				Expression expanded = rd.expandSymbols(section.get(i), vars);
				if ( expanded == null )
					expanded = new RangeExpression(
						new InfExpression(-1), new InfExpression(1));
				section.set(i, expanded);
			}
	}


	/**
	 * Expand every section under the constraints given by the range domain.
	 *
	 * @param rd the given range domain.
	 * @param ivs the set of symbols to be expanded.
	 * @param vars the set of symbols that should not be part of the expansion.
	 */
	public void expandMust(RangeDomain rd, Set<Symbol> ivs, Set<Symbol> vars)
	{
		Iterator<ELEMENT> iter = iterator();
		while ( iter.hasNext() )
		{
			ELEMENT section = iter.next();

			for ( int i=0; i<dimension; ++i )
			{
				Expression expanded = rd.expandSymbols(section.get(i), ivs);

				if ( expanded == null ||
				expanded instanceof RangeExpression &&
				!((RangeExpression)expanded).isBounded() ||
				Tools.containsSymbols(expanded, vars) )
				{
					iter.remove();
					break;
				}

				section.set(i, expanded);
			}
		}
	}


	/**
	 * Section intersection.
	 * @param other the section being intersected with.
	 * @param rd the supporting range domain.
	 */
	public Section intersectWith(Section other, RangeDomain rd)
	{
		// No intersection is possible; returns null (check at the higher level)
		if ( dimension != other.dimension )
			return null;

		Tools.printStatus(this + " (^) " + other + " = ", 2);

		Section ret = new Section(dimension);

		for ( ELEMENT section1 : this )
		{
			for ( ELEMENT section2 : other )
			{
				ELEMENT intersected = section1.intersectWith(section2, rd);

				if ( intersected != null )
					ret.add(intersected);
			}

			if ( ret.isEmpty() )
				break;
		}

		Tools.printlnStatus(ret.toString(), 2);
		//Tools.printlnStatus(this + " (^) " + other + " = " + ret, 2);

		return ret;
	}


	/**
	 * Section union.
	 * @param other the section being unioned with.
	 * @param rd the supporting range domain.
	 */
	public Section unionWith(Section other, RangeDomain rd)
	{
		if ( dimension != other.dimension )
			return null;

		Section ret = new Section(dimension);

		Iterator<ELEMENT> iter1 = iterator();
		Iterator<ELEMENT> iter2 = other.iterator();

		while ( iter1.hasNext() || iter2.hasNext() )
		{
			if ( !iter1.hasNext() )
				ret.add((ELEMENT)iter2.next().clone());

			else if ( !iter2.hasNext() )
				ret.add((ELEMENT)iter1.next().clone());

			else
			{
				ELEMENT section1 = iter1.next(), section2 = iter2.next();

				ELEMENT unioned = section1.unionWith(section2, rd);

				if ( unioned == null ) // union was not merged
				{
					ret.add((ELEMENT)section1.clone());
					ret.add((ELEMENT)section2.clone());
				}
				else                   // union was merged
					ret.add(unioned);
			}
		}

		Tools.printlnStatus(this + " (v) " + other + " = " + ret, 2);

		return ret;
	}


	/**
	 * Section difference.
	 *
	 * @param other the other section from which this section is differenced.
	 * @param rd the supporting range domain.
	 * @return the resulting section.
	 */
	public Section differenceFrom(Section other, RangeDomain rd)
	{
		Tools.printStatus(this + " (-) " + other + " = ", 2);

		Section ret = (Section)clone();

		// Just return a clone upon dimension mismatch
		if ( dimension != other.dimension )
		{
			Tools.printlnStatus(ret.toString(), 2);
			return ret;
		}

		for ( ELEMENT section2 : other )
		{
			Section curr = new Section(dimension);

			for ( ELEMENT section1 : ret )
			{
				Section diffed = section1.differenceFrom(section2, rd);

				for ( ELEMENT section : diffed )
					//if ( !curr.contains(section) )
					curr.add(section);
			}

			ret = curr;
		}

		Tools.printlnStatus(this + " (-) "+other + " = " + ret, 2);

		return ret;
	}


	/**
	 * Returns union of two symbolic bounds
	 */
	private static Expression unionBound
	(Expression e1, Expression e2, RangeDomain rd)
	{
		Expression intersected = intersectBound(e1, e2, rd);

		//System.out.println("intersected = "+intersected);

		if ( intersected == null ) // Either it has no intersection or unknown.
			return null;             // Merging i,i+1 => i:i+1 disregarded for now.

		RangeExpression re1 = RangeExpression.toRange(e1);
		RangeExpression re2 = RangeExpression.toRange(e2);

		Expression lb = null, ub = null;

		Relation rel = rd.compare(re1.getLB(), re2.getLB());

		if ( rel.isLE() )
			lb = re1.getLB();
		else if ( rel.isGE() )
			lb = re2.getLB();
		else
			return null;

		rel = rd.compare(re1.getUB(), re2.getUB());

		if ( rel.isGE() )
			ub = re1.getUB();
		else if ( rel.isLE() )
			ub = re2.getUB();
		else
			return null;

		return (new RangeExpression(lb, ub)).toExpression();
	}


	/**
	 * Returns intersection of two symbolic intervals
	 */
	private static Expression intersectBound
	(Expression e1, Expression e2, RangeDomain rd)
	{
		RangeExpression re1 = RangeExpression.toRange(e1);
		RangeExpression re2 = RangeExpression.toRange(e2);

		Expression lb = null, ub = null;

		//System.out.println("re1="+re1+",re2="+re2);

		Relation rel = rd.compare(re1.getLB(), re2.getLB());

		if ( rel.isGE() )
			lb = re1.getLB();
		else if ( rel.isLE() )
			lb = re2.getLB();
		else
			return null;

		rel = rd.compare(re1.getUB(), re2.getUB());

		if ( rel.isLE() )
			ub = re1.getUB();
		else if ( rel.isGE() )
			ub = re2.getUB();
		else
			return null;

		// Final check if lb>ub.
		rel = rd.compare(lb, ub);

		//System.out.println("comparing "+lb + " and " + ub +"\nunder"+rd);

		if ( !rel.isLE() )
			return null;
		else
			return (new RangeExpression(lb, ub)).toExpression();
	}


	/**
	 * Removes section elements that contain the specified variable.
	 */
	public void removeAffected(Symbol var)
	{
		Iterator<ELEMENT> iter = iterator();
		while ( iter.hasNext() )
		{
			boolean kill = false;

			for ( Expression e : iter.next() )
			{
				if ( Tools.containsSymbol(e, var) )
				{
					kill = true;
					break;
				}
			}

			if ( kill )
				iter.remove();
		}
	}


	/**
	 * Removes section elements that is affected by the specified function call.
	 */
	public void removeSideAffected(FunctionCall fc)
	{
		Set<Symbol> params = Tools.getAccessedSymbols(fc);
		Iterator<ELEMENT> iter = iterator();
		while ( iter.hasNext() )
		{
			boolean kill = false;

			for ( Expression e : iter.next() )
			{
				Set<Symbol> vars = Tools.getAccessedSymbols(e);
				vars.retainAll(params);
				// Case 1: variables in section representation are used as parameters.
				if ( !vars.isEmpty() )
				{
					kill = true;
					break;
				}
				// Case 2: variables in section representation are global.
				for ( Symbol var : Tools.getAccessedSymbols(e) )
				{
					if ( Tools.isGlobal(var, fc) )
					{
						kill = true;
						break;
					}
				}
				if ( kill )
					break;
			}
			if ( kill )
				iter.remove();
		}
	}

	/**
	 * Converts this section to a string.
	 *
	 * @return the string representation of the section.
	 */
	public String toString()
	{
		return ( "{" + Tools.listToString(this, ", ") + "}" );
	}

	/**
	 * Represents the elements contained in a section.
	 */
	public static class ELEMENT extends ArrayList<Expression>
	implements Cloneable
	{

/**
 * Constructs a new element.
 */
		public ELEMENT()
		{
			super();
		}

/**
 * Constructs a new element from the given array access.
 */
		public ELEMENT(ArrayAccess acc)
		{
			for ( int i=0; i < acc.getNumIndices(); ++i )
				add((Expression)acc.getIndex(i).clone());
		}

/**
 * Returns a clone of this element.
 */
		public Object clone()
		{
			ELEMENT o = new ELEMENT();

			for ( Expression e : this )
				o.add((Expression)e.clone());

			return o;
		}

/**
 * Checks if this element is equal to the specified object.
 */
		public boolean equals(Object o)
		{
			if ( o == null || o.getClass() != this.getClass() )
				return false;

			ELEMENT other = (ELEMENT)o;

			if ( size() != other.size() )
				return false;

			for ( int i=0; i < size(); ++i )
			{
				if ( !get(i).equals(other.get(i)) )
					return false;
			}

			return true;
		}

/**
 * Converts this element to a string.
 *
 * @return the string representation of this element.
 */
		public String toString()
		{
			StringBuilder str = new StringBuilder(80);

			str.append("[");

			Iterator<Expression> iter = iterator();

			if ( iter.hasNext() )
				str.append(iter.next().toString().replaceAll("\\[|\\]", ""));

			while ( iter.hasNext() )
				str.append("]["+iter.next().toString().replaceAll("\\[|\\]", ""));

			str.append("]");

			return str.toString();
		}

/**
 * Intersects this element with another element under the given range domain.
 *
 * @param other the other element.
 * @param rd the specified range domain.
 * @return the result of the intersection.
 */
		public ELEMENT intersectWith(ELEMENT other, RangeDomain rd)
		{
			ELEMENT ret = new ELEMENT();

			for ( int i=0; i < size(); ++i )
			{
				Expression intersected = intersectBound(get(i), other.get(i), rd);

				if ( intersected == null ) // Either it is empty or unknown
					return null;

				ret.add(intersected);
			}

			return ret;
		}

/**
 * Union this element with another element under the given range domain.
 *
 * @param other the other element.
 * @param rd the specified range domain.
 * @return the result of the union.
 */
		public ELEMENT unionWith(ELEMENT other, RangeDomain rd)
		{
			ELEMENT ret = new ELEMENT();

			for ( int i=0; i < size(); ++i )
			{
				Expression unioned = unionBound(get(i), other.get(i), rd);

				if ( unioned == null ) // Either it has holes or unknown
					return null;

				ret.add(unioned);
			}

			return ret;
		}

/**
 * Differences this element from another element under the given range domain.
 *
 * @param other the other element.
 * @param rd the specified range domain.
 * @return the resulting section of the difference.
 */
		public Section differenceFrom(ELEMENT other, RangeDomain rd)
		{
			// Temporary list containing the result of differences for each dimension
			Section ret = new Section(size());

			for ( int i=0; i < size(); ++i )
			{
				List<Expression> temp_i = new ArrayList<Expression>();

				Expression intersected = intersectBound(get(i), other.get(i), rd);

				//System.out.println("intersected="+intersected);

				if ( intersected == null )
					temp_i.add((Expression)get(i).clone());

				else
				{
					RangeExpression re_inct = RangeExpression.toRange(intersected);
					RangeExpression re_from = RangeExpression.toRange(get(i));
					Expression one = new IntegerLiteral(1);

					Expression left_ub = NormalExpression.subtract(re_inct.getLB(),one);
					Expression right_lb = NormalExpression.add(re_inct.getUB(), one);

					Relation rel = rd.compare(re_from.getLB(), left_ub);

					if ( !rel.isGT() )
						temp_i.add(
						(new RangeExpression((Expression)re_from.getLB().clone(), left_ub))
						.toExpression());

					rel = rd.compare(right_lb, re_from.getUB());

					if ( !rel.isGT() )
						temp_i.add(
						(new RangeExpression(right_lb, (Expression)re_from.getUB().clone()))
						.toExpression());
				}

				for ( Expression e : temp_i )
				{
					ELEMENT new_section = (ELEMENT)clone();
					new_section.set(i, e);
					ret.add(new_section);
				}
			}

			return ret;
		}

	}


/**
 * Class MAP represents map from variables to their sections. For the
 * convenience of implementation, we assign empty section for scalar variables.
 */
	public static class MAP extends HashMap<Symbol,Section> implements Cloneable
	{

/**
 * Constructs an empty map.
 */
		public MAP()
		{
			super();
		}

/**
 * Constructs a map with a pair of variable and section.
 */
		public MAP(Symbol var, Section section)
		{
			super();
			put(var, section);
		}

/**
 * Clone method
 */
		public Object clone()
		{
			MAP o = new MAP();

			for ( Symbol var : keySet() )
				o.put(var, (Section)get(var).clone());

			return o;
		}

/**
 * Clean empty sections
 */
		public void clean()
		{
			Set<Symbol> vars = new HashSet<Symbol>(keySet());

			for ( Symbol var : vars )
				if ( get(var).dimension > 0 && get(var).isEmpty() )
					remove(var);
		}

/**
 * Intersection operation
 */
		public MAP intersectWith(MAP other, RangeDomain rd)
		{
			MAP ret = new MAP();

			if ( other == null )
				return ret;

			for ( Symbol var : keySet() )
			{
				Section s1 = get(var);
				Section s2 = other.get(var);

				if ( s1 == null || s2 == null )
					continue;

				if ( s1.isScalar() && s2.isScalar() )
					ret.put(var, (Section)s1.clone());

				//else if ( !s1.isScalar() && !s2.isScalar() )
				else
				{
					Section intersected = s1.intersectWith(s2, rd);

					if ( intersected == null )
						Tools.printlnStatus("[WARNING] Dimension mismatch", 0);

					else
						ret.put(var, intersected);
				}
			}

			ret.clean();
			return ret;
		}

/**
 * Union operation
 */
		public MAP unionWith(MAP other, RangeDomain rd)
		{
			if ( other == null )
				return (MAP)clone();

			MAP ret = new MAP();

			Set<Symbol> vars = new HashSet<Symbol>(keySet());
			vars.addAll(other.keySet());

			for ( Symbol var : vars )
			{
				Section s1 = get(var);
				Section s2 = other.get(var);

				if ( s1 == null && s2 == null )
					continue;

				if ( s1 == null )
					ret.put(var, (Section)s2.clone());
				else if ( s2 == null )
					ret.put(var, (Section)s1.clone());
				else if ( s1.isScalar() && s2.isScalar() )
					ret.put(var, (Section)s1.clone());
				else
				{
					Section unioned = s1.unionWith(s2, rd);

					if ( unioned == null )
						ret.put(var, (Section)s2.clone()); // heuristics -- second operand
					else
						ret.put(var, unioned);
				}
			}

			ret.clean();
			return ret;
		}

/**
 * Difference operation
 */
		public MAP differenceFrom(MAP other, RangeDomain rd)
		{
			if ( other == null )
				return (MAP)clone();

			MAP ret = new MAP();

			Set<Symbol> vars = new HashSet<Symbol>(keySet());

			for ( Symbol var : vars )
			{
				Section s1 = get(var);
				Section s2 = other.get(var);

				if ( s2 == null )
					ret.put(var, (Section)s1.clone());

				//else if ( !s1.isScalar() )
				else if ( s1.isArray() || s2.isArray() )
					ret.put(var, s1.differenceFrom(s2, rd));
			}

			ret.clean();
			return ret;
		}

/**
 * Removes sections that contains the specified symbol.
 */
		public void removeAffected(Symbol var)
		{
			Set<Symbol> keys = new HashSet<Symbol>(keySet());

			for ( Symbol key : keys )
				get(key).removeAffected(var);

			clean();
		}

/**
 * Removes sections that contains the specified set of variables.
 */
		public void removeAffected(Collection<Symbol> vars)
		{
			for ( Symbol var : vars )
				removeAffected(var);
		}

/**
 * Removes sections that are unsafe in the given traversable object due to
 * function calls.
 */
		public void removeSideAffected(Traversable tr)
		{
			DepthFirstIterator iter = new DepthFirstIterator(tr);

			iter.pruneOn(FunctionCall.class);

			while ( iter.hasNext() )
			{
				Object o = iter.next();
				if ( o instanceof FunctionCall )
				{
					Set<Symbol> vars = new HashSet<Symbol>(keySet());

					for ( Symbol var : vars )
						get(var).removeSideAffected((FunctionCall)o);

					clean();
				}
			}
		}

	}

}
