package cetus.analysis;

import java.util.*;
import cetus.hir.*;

/**
 * NormalExpression supports normalization and simplification of expressions.
 * It internally transforms a given expression to a normalized form so that
 * each operand of a commutative and associative operation is located at the
 * same height in the tree representation; in other words, it converts a binary
 * tree to a n-ary tree. This internal representation makes the simplification
 * algorithm easier and the result of the simplification is converted back to
 * the original internal representation of Cetus.
 */
public class NormalExpression implements Comparable,Comparator<NormalExpression>
{
  // Type information
  private static final int INTEGER_ARITHMETIC = 1;
  private static final int FLOAT_ARITHMETIC = 2;

  // Expression type that is not handled by the normalizer.
  private static final int OMEGA = 999;

  // Frequently used common objects.
  private static final Expression one = new IntegerLiteral(1);
  private static final Expression zero = new IntegerLiteral(0);
  private static final NormalExpression One = new NormalExpression(one);
  private static final NormalExpression Zero = new NormalExpression(zero);
  
	// Normalized operator.
  private OP op;           

	// Arithmetic type.
  private int type;        

	// Symbolic cardinality of the normal expression.
  private int order;       

	// List of children
  private ArrayList list;  

  /**
   * Returns a normalized(simplified) expression of an {@link Expression}.
   * It is important to notice that this method returns a normalized form and
   * this transformation may or may not result in reduced complexity in the
   * expression. Normalized expressions can be useful in analysis passes
   * by imposing a certain orders in the expression. Pass writers may be able
   * to customize the normalization process with
   * {@link #simplify(Expression,String)}.
   *
   * @param e the {@link Expression} object to be normalized.
   * @return  the normalized form of the expression e.
   * @see     #simplify(Expression, String)
   */
  public static Expression simplify(Expression e)
  {
    if ( isNotSimplifiable(e) )
      return e;
    else
      return simplify(e, "SIMP");
  }

  /**
   * Returns a normalized expression of an {@link Expression} after applying
   * a specific option. These options include:
   *
   * <ul>
   * <li>"FOLD" , expression folding, e.g. 1-2+a+2*a -> -1+3*a
   * <li>"DIST" , expression distribution, e.g. 2*(a+b) -> 2*a+2*b
   * <li>"DIV"  , expression division/mod, e.g. a*b/a -> b
   * <li>"LOGIC", logical expression simplification
   * </ul>
   *
   * @param e   the {@link Expression} object to be normalized.
   * @param opt the normalization option.
   * @return    the normalized form of the expression e.
   */
  public static Expression simplify(Expression e, String opt)
  {
    Expression ret = (Expression)e.clone();

    simpFromBottom(ret, opt);

    if ( !Tools.containsSideEffect(ret) )
    {
      ret = simpTemplates(ret);
      if ( opt.equals("RANGE") ) opt = "SIMP";
      ret = (new NormalExpression(ret)).simplify(opt).getExpression();
    }

    return ret;
  }

	/**
	 * Compares the two expressions based on their normal forms.
	 *
	 * @param e1 the first expression.
	 * @param e2 the second expression.
	 * @return 0 if they are equal, -1 if e1 contains less symbolic variables,
	 * 1 otherwise.
	 */
  public static int compare(Expression e1, Expression e2)
  {
    NormalExpression ne1 = new NormalExpression(e1);
    NormalExpression ne2 = new NormalExpression(e2);
    //return ne1.compare(ne2);
    return ne1.compareTo(ne2);
  }

  /**
   * Returns addition of the two expressions with simplification.
   *
   * @param e1 the first operand of addition.
   * @param e2 the second operand of addition.
   * @return   the result of addition.
   */
  public static Expression add(Expression e1, Expression e2)
  {
    Expression e = new BinaryExpression(e1, BinaryOperator.ADD, e2);
    return simplify(e, "SIMP");
  }

  /**
   * Returns subtraction of two expressions with simplification.
   *
   * @param e1 the first operand of subtraction.
   * @param e2 the second operand of subtraction.
   * @return   the result of subtraction.
   */
  public static Expression subtract(Expression e1, Expression e2)
  {
    Expression e = new BinaryExpression(e1, BinaryOperator.SUBTRACT, e2);
    return simplify(e, "SIMP");
  }

  /**
   * Returns multiplication of two expressions with simplification.
   *
   * @param e1 the first operand of multiplication.
   * @param e2 the second operand of multiplication.
   * @return   the result of multiplication.
   */
  public static Expression multiply(Expression e1, Expression e2)
  {
    Expression e = new BinaryExpression(e1, BinaryOperator.MULTIPLY, e2);
    return simplify(e, "SIMP");
  }

  /**
   * Returns division of two expressions with simplification.
   *
   * @param e1 the first operand of division.
   * @param e2 the second operand of division.
   * @return   the result of division.
   */
  public static Expression divide(Expression e1, Expression e2)
  {
    Expression e = new BinaryExpression(e1, BinaryOperator.DIVIDE, e2);
    return simplify(e, "SIMP");
  }

	/**
	 * Returns a logically negated expression of the given expression.
	 *
	 * @param e the input expression.
	 * @return the negated expression.
	 */
  public static Expression negate(Expression e)
  {
    Expression e1 = new UnaryExpression(UnaryOperator.LOGICAL_NEGATION, e);
    return simplify(e1, "SIMP");
  }

	/**
	 * Returns an equality expression e1==e2 with simplification.
	 *
	 * @param e1 the first expression.
	 * @param e2 the second expression.
	 * @return the resulting expression.
	 */
  public static Expression eq(Expression e1, Expression e2)
  {
    Expression e = new BinaryExpression(e1, BinaryOperator.COMPARE_EQ, e2);
    return simplify(e, "SIMP");
  }

	/**
	 * Returns a binary expression e1&&e2 with simplification.
	 *
	 * @param e1 the first expression.
	 * @param e2 the second expression.
	 * @return the resulting expression.
	 */
	public static Expression and(Expression e1, Expression e2)
	{
    Expression e = new BinaryExpression(e1, BinaryOperator.LOGICAL_AND, e2);
    return simplify(e, "SIMP");
	}

	/**
	 * Returns a binary expression e1||e2 with simplification.
	 *
	 * @param e1 the first expression
	 * @param e2 the second expression
	 * @return the resulting expression
	 */
	public static Expression or(Expression e1, Expression e2)
	{
    Expression e = new BinaryExpression(e1, BinaryOperator.LOGICAL_OR, e2);
    return simplify(e, "SIMP");
	}

	/**
	 * Returns the string format of the normal expression (for debugging).
	 *
	 * @return the string format of the normal expression.
	 */
  public String toString()
  {
    if ( op.equals(OP.UNIT) )
      return "("+order+":"+list.get(0)+")";
    String ret = "(";
    ret += order+":"+op.getOP().toString();
    for ( Object o : list )
      ret += ","+o.toString();
    return ret+")";
  }

	/*
	 * Compares this normal expression to the specified object.
	 *
	 * @param o the object compared to.
	 * @return the integer result of the comparison.
	 */
	public int compareTo(Object o)
	{
		if ( o == null )
			return 1;

		if ( o instanceof Expression )
			return getExpression().toString().compareTo(o.toString());

		if ( o instanceof NormalExpression )
			return compare(this, (NormalExpression)o);

		return 1;
	}

	/*
	 * Checks if this normal expression is equal to the specified object.
	 *
	 * @param o the object compared to.
	 * @return true if they are normalized to the same expression, false
	 * otherwise.
	 */
	public boolean equals(Object o)
	{
		return (compareTo(o) == 0);
	}

	/*
	 * Compares tow normal expressions; {@link Comparator} requires this method.
	 *
	 * @param re1 the first normal expression.
	 * @param re2 the second normal expression.
	 * @return the integer result of the comparison.
	 */
	public int compare(NormalExpression re1, NormalExpression re2)
	{
		if ( re1.order < re2.order )
			return -1;

		if ( re1.order > re2.order )
			return 1;

		NormalExpression var1 = re1.getVariable(), var2 = re2.getVariable();
		Expression e1 = re1.getExpression(), e2 = re2.getExpression();

		if ( var1 != null && var2 != null &&
		!(e1.equals(var1.getExpression()) && e2.equals(var2.getExpression())) )
			return compare(var1, var2);

		if ( re1.list.size() < re2.list.size() )
			return -1;

		if ( re1.list.size() > re2.list.size() )
			return 1;

		return e1.toString().compareTo(e2.toString());
	}

	// Returns a normal expression object for direct access.
  protected static NormalExpression normalSimplify(Expression e)
  {
    NormalExpression ret = new NormalExpression(e);
    return ret.simplify("SIMP");
  }

  // Return the constant term in the simplified expression.
  // The return value is non-zero if op is ADD and the expression contains a
  // non-symbolic term.
  protected long getConstantCoefficient()
  {
    if ( isLiteral() )
      return toDouble().intValue();

    if ( !op.equals(OP.ADD) )
      return 0;
    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.isLiteral() )
        return curr.toDouble().longValue();
    }
    return 0;
  }

  // Return the coefficient multiplied by the identifier
  protected long getCoefficient(Identifier id)
  {
    if ( compareTo(id) == 0 )
      return 1;

    if ( op.equals(OP.MUL) && getVariable().compareTo(id) == 0 )
      return getCoef().toDouble().intValue();

    if ( !op.equals(OP.ADD) )
      return 0;

    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.getVariable() == null )
        continue;
      if ( curr.getVariable().compareTo(id) == 0 )
        return curr.getCoef().toDouble().longValue();
    }
    return 0;
  }

  // Return the list of symbolic terms that are instance of Identifier.
  // If there is any other type of symbolic terms just return null, indicating
  // the expression cannot be affine.
  protected List<Identifier> getVariableList()
  {
    List<Identifier> ret = new ArrayList<Identifier> ();

    switch ( op )
    {
      case UNIT:
      {
        Expression e = getExpression();
        if ( e instanceof Identifier )
          ret.add((Identifier)e);
        else if ( !isLiteral() )
          return null;
        break;
      }
      case MUL:
      {
        NormalExpression var = getVariable();
        //System.out.println(var.getExpression());
        if ( var != null && var.getExpression() instanceof Identifier )
          ret.add((Identifier)var.getExpression());
        else
          return null;
        break;
      }
      case ADD:
        for ( Object o: list )
        {
          List<Identifier> curr = ((NormalExpression)o).getVariableList();
          if ( curr == null )
            return null;
          ret.addAll(curr);
        }
        break;
      default:
        return null;
    }
    return ret;
  }

  // Return whether this expression is affine w.r.t. the given list
  protected boolean isAffine(List<Identifier> id_list)
  {
    List<Identifier> my_list = getVariableList();
    if ( my_list==null || id_list==null )
      return false;
    // FIXME: can be replaced by symbol comparison in the future
    // For me to be affine, all my ids should appear in id_list
    //
    // With Symbol interface
    /*
    Set<Symbol> symbols = new HashSet<Symbol>();
    for ( Identifier id : id_list )
      symbols.add(id.getSymbol());
    for ( Identifier my_id : my_list )
      if ( !symbols.contains(my_id.getSymbol()) )
        return false;
    return true;
    */
    for ( Identifier my_id : my_list )
    {
      boolean my_id_exist = false;
      for ( Identifier id : id_list )
      {
        if ( my_id.compareTo(id) == 0 )
        {
          my_id_exist = true;
          break;
        }
      }
      if ( !my_id_exist )
        return false;
    }
    return true;
  }

  // Return a set of equivalent expressions that contain at most one
  // identifier on the left hand side.
  protected static List solveForVariables(Expression e)
  {
    List ret = new ArrayList();
    NormalExpression ne = normalSimplify(e);

    if ( ne.op.equals(OP.AND) || ne.op.equals(OP.OR) )
    {
      for ( Object o : ne.list )
        ret.add(solveForVariables(((NormalExpression)o).getExpression()));
      return ret;
    }
    else if ( !ne.op.isCompare() )
      return ret;

    // Normalized comparison should contain 0 on the rhs.
    NormalExpression lhs = ne.get(0), rhs = ne.get(1);
    Expression lhs_e = lhs.getExpression();

    if ( lhs_e instanceof Identifier &&
      !Tools.containsExpr(rhs.getExpression(), lhs_e) )
      ret.add(ne.getExpression());

    else if ( lhs.op.equals(OP.ADD) )
    {
      // 1st phase with the original expression
      for ( Object o : lhs.list )
      {
        Expression curr = ((NormalExpression)o).getExpression();
        if ( curr instanceof Identifier )
        {
          NormalExpression others = Zero;
          for ( Object oo : lhs.list )
          {
            if ( oo != o )
              others = subtract(others, (NormalExpression)oo);
          }
          Expression others_e = others.getExpression();
          if ( !Tools.containsExpr(others_e, curr) )
            ret.add(new BinaryExpression(
              curr, (BinaryOperator)ne.op.getOP(), others_e));
        }
      }

      // 2nd phase with exchanged expression
      NormalExpression elhs = normalSimplify(new BinaryExpression(
        new IntegerLiteral(-1), BinaryOperator.MULTIPLY, lhs.getExpression()));
      for ( Object o : elhs.list )
      {
        Expression curr = ((NormalExpression)o).getExpression();
        if ( curr instanceof Identifier )
        {
          NormalExpression others = Zero;
          for ( Object oo : elhs.list )
          {
            if ( oo != o )
              others = subtract(others, (NormalExpression)oo);
          }
          Expression others_e = others.getExpression();
          if ( !Tools.containsExpr(others_e, curr) )
          ret.add(new BinaryExpression(
            curr, (BinaryOperator)ne.op.getExchange().getOP(), others_e));
        }
      }
    }
    return ret;
  }


	// Empty constructor
  private NormalExpression()
  {
    op = OP.UNIT;
    type = INTEGER_ARITHMETIC;
    order = 0;
    list = new ArrayList();
  }

	// Constructor with an expression
  private NormalExpression(Expression e)
  {
    this();
    if ( e instanceof UnaryExpression )
      parseUnary((UnaryExpression)e);
    else if ( e instanceof BinaryExpression )
      parseBinary((BinaryExpression)e);
    else
      parseOther(e);
    sort();
  }

	// Constructor with an operator.
  private NormalExpression(OP op)
  {
    this();
    this.op = op;
  }

	// Returns a normal expression for an integer literal.
  private static NormalExpression getInteger(int num)
  {
    return new NormalExpression(new IntegerLiteral(num));
  }

	// Returns a normal expression for a float literal.
  private static NormalExpression getFloat(double num)
  {
    return new NormalExpression(new FloatLiteral(num));
  }

  // Adds a child in the list.
  private void add(NormalExpression ne)
  {
    list.add(ne);
    order += ne.order;
  }

  // Adds a child and merge if necessary.
  private void addMerge(NormalExpression ne)
  {
    if ( op.equals(ne.op) && op.isCommAssoc() )
      list.addAll(ne.list);
    else
      list.add(ne);
    order += ne.order;
  }

  // Joins two normal expressions.
  private static NormalExpression join
  (NormalExpression ne1, OP op, NormalExpression ne2)
  {
    NormalExpression ret = new NormalExpression(op);
    ret.addMerge(ne1);
    ret.addMerge(ne2);
    return ret;
  }

  // Set a child in the list.
  private void set(int loc, NormalExpression ne)
  {
    order += ne.order - get(loc).order;
    list.set(loc, ne);
  }

  // Returns the loc-th child in the list.
  private NormalExpression get(int loc)
  {
    return (NormalExpression)list.get(loc);
  }

  // Checks if the normal expression is literal.
  private boolean isLiteral()
  {
    return ( isInteger() || isFloat() );
  }

	// Checks if the normal expression is integer.
  private boolean isInteger()
  {
    return ( list.get(0) instanceof IntegerLiteral );
  }

	// Checks if the normal expression is float.
  private boolean isFloat()
  {
    return ( list.get(0) instanceof FloatLiteral );
  }

  // Returns double value of a literal expression
  private Double toDouble()
  {
    return new Double(list.get(0).toString());
  }

  // Sorts the child expressions if reordering is safe using the comparator
  private void sort()
  {
    if ( list.size() < 2 )
      return;
    for ( Object o : list )
      ((NormalExpression)o).sort();
    if ( op.isCommAssoc() )
      //Collections.sort(list, new NormalComparator());
      Collections.sort(list, this);
  }

  // Match and build a unary expression
  private void parseUnary(UnaryExpression e)
  {
    UnaryOperator eop = e.getOperator();
    Expression esub = e.getExpression();

    if ( eop == UnaryOperator.LOGICAL_NEGATION )
    {
      op = OP.NEG;
      this.add(new NormalExpression(esub));
    }
    else if ( eop == UnaryOperator.MINUS )
    {
      if ( esub instanceof IntegerLiteral )
        list.add(new IntegerLiteral(-1*((IntegerLiteral)esub).getValue()));
      else if ( esub instanceof FloatLiteral )
        list.add(new FloatLiteral(-1*((FloatLiteral)esub).getValue()));
      else
      {
        op = OP.MUL;
        list.add(new NormalExpression(new IntegerLiteral(-1)));
        NormalExpression sub = new NormalExpression(esub);
        if ( op.equals(sub.op) )
          list.addAll(sub.list);
        else
          list.add(sub);
        order = sub.order;
      }
    }
    else if ( eop == UnaryOperator.PLUS )
    {
      if ( esub instanceof IntegerLiteral || esub instanceof FloatLiteral )
        list.add(esub);
      else
      {
        NormalExpression sub = new NormalExpression(esub);
        op = sub.op; order = sub.order; type = sub.type;
        list.clear(); list.addAll(sub.list);
      }
    }
    else
    {
      list.add(e);
      order = OMEGA;
    }
  }

  // Match and build a binary expression
  private void parseBinary(BinaryExpression e)
  {
    BinaryOperator eop = e.getOperator();
    NormalExpression lsub = new NormalExpression(e.getLHS());
    NormalExpression rsub;
    Expression erhs = e.getRHS();
    op = OP.setOP(eop);

    if ( eop == BinaryOperator.SUBTRACT )
    {
      op = OP.ADD;
      erhs = new BinaryExpression(
        new IntegerLiteral(-1), BinaryOperator.MULTIPLY, erhs);
    }

    if ( op.equals(OP.UNIT) )
    {
      list.add(e);
      order = OMEGA;
      return;
    }

    rsub = new NormalExpression(erhs);

    // Merge associative/commutative operations to the same tree level
    addMerge(lsub);
    addMerge(rsub);
    
    // Compute expression's order
    order = lsub.order+rsub.order;
    if ( order > OMEGA ) order = OMEGA;

  }

  // Match and build other expressions
  private void parseOther(Expression e)
  {
    list.add(e);
    if ( e instanceof IntegerLiteral || e instanceof FloatLiteral )
      order = 0;
    else if ( e instanceof Identifier )
      order = 1;
    else
      order = OMEGA;
  }

  // Rebuild C expression from a normal expression
  private Expression getExpression()
  {
    Expression e = null;
    
    if ( op.equals(OP.UNIT) )
      e = (Expression)((Expression)list.get(0)).clone();
    else if ( op.equals(OP.NEG) )
      e = new UnaryExpression(
        UnaryOperator.LOGICAL_NEGATION ,get(0).getExpression());
    else {
      for ( Object o: list )
      {
        NormalExpression curr = (NormalExpression)o;
        if ( e == null )
          e = curr.getExpression();
        else
          e = new BinaryExpression(
            e, (BinaryOperator)op.getOP(), curr.getExpression());
      }
    }
    return e;
  }

  // Operations +,-,*,/ with normal expressions
  private static NormalExpression add
  (NormalExpression e1, NormalExpression e2)
  {
    return compute(e1, OP.ADD, e2);
  }

  private static NormalExpression subtract
  (NormalExpression e1, NormalExpression e2)
  {
    return compute(e1, OP.ADD,
      compute(getInteger(-1), OP.MUL, e2).dist("SIMP"));
  }

  private static NormalExpression multiply
  (NormalExpression e1, NormalExpression e2)
  {
    return compute(e1, OP.MUL, e2);
  }

  private static NormalExpression divide
  (NormalExpression e1, NormalExpression e2)
  {
    return compute(e1, OP.DIV, e2);
  }

  private static NormalExpression compute
  (NormalExpression e1, OP op, NormalExpression e2)
  {
    if ( e1.isLiteral() && e2.isLiteral() )
    {
      Double result = null;
      switch (op)
      {
        case ADD: result = e1.toDouble()+e2.toDouble(); break;
        case MUL: result = e1.toDouble()*e2.toDouble(); break;
        case DIV: result = e1.toDouble()/e2.toDouble(); break;
        default: return null;
      }
      if ( e1.isInteger() && e2.isInteger() )
        return getInteger(result.intValue());
      else
        return getFloat(result.doubleValue());
    }
    NormalExpression ret = new NormalExpression(op);
    ret.addMerge(e1);
    ret.addMerge(e2);
    return ret.fold("FOLD");
  }

  // Normalizes relation expressions by sending all terms to the left.
  // The RHS is always zero.
  private NormalExpression NormalRelation()
  {
    if ( !op.isCompare() )
      return this;

    NormalExpression ret = new NormalExpression(op);

    //if ( get(0).compare(get(1)) < 0 )
    if ( compare(get(0), get(1)) < 0 )
    {
      ret.add(subtract(get(1), get(0)));
      ret.op = ret.op.getExchange();
    }
    else
      ret.add(subtract(get(0), get(1)));
    
    ret.add(getInteger(0));
    return ret;
  }

  // Returns the numeric coefficient of the expression.
  // Assumes the expression is normalized.
  private NormalExpression getCoef()
  {
    NormalExpression ret = getInteger(1);

    if ( !op.equals(OP.UNIT) && !op.equals(OP.MUL) )
      return ret;

    if ( op.equals(OP.UNIT) )
      return (isLiteral())? this: ret;

    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( !curr.isLiteral() )
        return ret;
      ret = multiply(ret, curr);
    }
    return ret;
  }

  // Returns the symbolic part of the expression.
  // Assumes the expression is normalized.
  private NormalExpression getVariable()
  {
    if ( isLiteral() )
      return null;

    if ( !op.equals(OP.MUL) )
      return this;

    NormalExpression ret = new NormalExpression(OP.MUL);
    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.order > 0 )
        ret.add(curr);
    }

    if ( ret.list.size() < 1 )
      return null;
    else
      return (ret.list.size()==1)? ret.get(0): ret;
  }

	// Returns a symbolic GCD of the two normal expressions.
  private static NormalExpression getGCD
  (NormalExpression ne1, NormalExpression ne2)
  {
    NormalExpression gcd = new NormalExpression(OP.MUL);
    NormalExpression div1 = new NormalExpression(OP.MUL);
    NormalExpression div2 = new NormalExpression(OP.MUL);
    // Normalize forms
    NormalExpression e1 = join(getInteger(1), OP.MUL, ne1);
    NormalExpression e2 = join(getInteger(1), OP.MUL, ne2);
    int num1 = 1, num2 = 1;
    for ( int i1=0,i2=0; i1<e1.list.size() || i2<e2.list.size(); )
    {
      NormalExpression curr1 = (i1<e1.list.size())? e1.get(i1): null;
      NormalExpression curr2 = (i2<e2.list.size())? e2.get(i2): null;
      // Remainders
      if ( curr1 == null && !curr2.isInteger() )
      {
        div2.add(curr2); i2++;
        continue;
      }
      if ( curr2 == null && !curr1.isInteger() )
      {
        div1.add(curr1); i1++;
        continue;
      }
      // Integer parts
      if ( curr1 != null && curr1.isInteger() )
      {
        num1 *= curr1.toDouble().intValue(); i1++;
        continue;
      }
      if ( curr2 != null && curr2.isInteger() )
      {
        num2 *= curr2.toDouble().intValue(); i2++;
        continue;
      }
      if ( curr1 == null || curr2 == null )
        continue;
      // Common expressions
      //int rel = curr1.compare(curr2);
      int rel = curr1.compareTo(curr2);
      if ( rel < 0 )       { div1.add(curr1); i1++; }
      else if ( rel == 0 ) { gcd.add(curr1); i1++; i2++; }
      else                 { div2.add(curr2); i2++; }
    }

    // Assemble numeric parts and symbolic parts
    int numGCD = cetus.analysis.GCD.compute(num1, num2);
    NormalExpression ngcd = getInteger(numGCD);
    NormalExpression ndiv1 = getInteger(num1/numGCD);
    NormalExpression ndiv2 = getInteger(num2/numGCD);
    gcd = (gcd.list.size()==0)? ngcd: join(ngcd, OP.MUL, gcd);
    div1 = (div1.list.size()==0)? ndiv1: join(ndiv1, OP.MUL, div1);
    div2 = (div2.list.size()==0)? ndiv2: join(ndiv2, OP.MUL, div2);

    // Pack the results in a list: here ret is used just as a list
    NormalExpression ret = new NormalExpression();
    ret.add(gcd.fold("FOLD"));
    ret.add(div1.fold("FOLD"));
    ret.add(div2.fold("FOLD"));
    return ret;
  }

  // Test if e is parsed to normal forms.
  private static boolean isNormal(Expression e)
  {
    if ( e instanceof Identifier ||
      e instanceof IntegerLiteral ||
      e instanceof FloatLiteral )
      return true;

    if ( e instanceof BinaryExpression )
    {
      Set op_set = new HashSet(Arrays.asList(
        "+","-","*","/","==",">=",">","<=","<","!=","&&","||","%"));
      return op_set.contains(((BinaryExpression)e).getOperator().toString());
    }

    else if ( e instanceof UnaryExpression )
    {
      Set op_set = new HashSet(Arrays.asList("!","-","+"));
      return op_set.contains(((UnaryExpression)e).getOperator().toString());
    }

    return false;
  }

  // Test if there is any factors that prevent reordering.
  private static boolean isNotSimplifiable(Expression e)
  {
    DepthFirstIterator iter = new DepthFirstIterator(e);
    while ( iter.hasNext() )
    {
      Object o = iter.next();
      if ( o instanceof BinaryExpression )
      {
        BinaryExpression be = (BinaryExpression)o;
    		// Template 1: A-B where B is address type
        if ( be.getOperator() == BinaryOperator.SUBTRACT &&
          be.getRHS() instanceof Identifier )
        {
          Symbol var = ((Identifier)be.getRHS()).getSymbol();
          if ( var == null ||
            var.getTypeSpecifiers().contains(PointerSpecifier.UNQUALIFIED) ||
            var.getTypeSpecifiers().contains(PointerSpecifier.CONST) ||
            var.getTypeSpecifiers().contains(PointerSpecifier.VOLATILE) ||
            var.getTypeSpecifiers().contains(PointerSpecifier.CONST_VOLATILE) )
            return true;
        }
      }
    }
    return false;
  }


  // Recursive simplification identifying each expression
  private static void simpFromBottom(Expression e, String opt)
  {
    FlatIterator iter = new FlatIterator(e);

    while ( iter.hasNext() )
    {
      Object child = iter.next();
      if ( !(child instanceof Expression) )
        return;
      simpFromBottom((Expression)child, opt);
    }

    if ( isNormal(e) && !opt.equals("RANGE"))
      return;

    for ( iter.reset(); iter.hasNext(); )
    {
      Expression child = (Expression)iter.next();
      if ( !Tools.containsSideEffect(child) )
      {
        Expression temp = simpTemplates(child);
        if ( opt.equals("RANGE") ) opt = "SIMP";
        NormalExpression ne = (new NormalExpression(temp)).simplify(opt);
        child.swapWith((ne.getExpression()));
      }
    }
  }

  // Templates for binary expression matching
  private static Expression binaryTemplates(BinaryExpression e)
  {

    Expression lhs = e.getLHS(), rhs = e.getRHS();

    BinaryOperator op = e.getOperator();

    Expression ret = e;

    if ( lhs instanceof IntegerLiteral && rhs instanceof IntegerLiteral )
    {
      long lhs_value = (new Long(lhs.toString())).longValue();
      long rhs_value = (new Long(rhs.toString())).longValue();

      if ( op == BinaryOperator.BITWISE_AND )
        ret = new IntegerLiteral(lhs_value & rhs_value);

      else if ( op == BinaryOperator.BITWISE_EXCLUSIVE_OR )
        ret = new IntegerLiteral(lhs_value ^ rhs_value);

      else if ( op == BinaryOperator.BITWISE_INCLUSIVE_OR )
        ret = new IntegerLiteral(lhs_value | rhs_value);

      else if ( op == BinaryOperator.SHIFT_LEFT )
        ret = new IntegerLiteral(lhs_value << rhs_value);

      else if ( op == BinaryOperator.SHIFT_RIGHT )
        ret = new IntegerLiteral(lhs_value >> rhs_value);
    }

    else if ( op == BinaryOperator.ADD )
    {
      if ( lhs instanceof MinMaxExpression )
      {
        ret = new MinMaxExpression(((MinMaxExpression)lhs).isMin());
        for ( Object o : lhs.getChildren() )
          ((MinMaxExpression)ret).add(
          NormalExpression.add((Expression)o, rhs));
        ret = minmaxTemplates((MinMaxExpression)ret);
      }
      else if ( rhs instanceof MinMaxExpression )
      {
        ret = new MinMaxExpression(((MinMaxExpression)rhs).isMin());
        for ( Object o : rhs.getChildren() )
          ((MinMaxExpression)ret).add(
          NormalExpression.add(lhs, (Expression)o));
        ret = minmaxTemplates((MinMaxExpression)ret);
      }
    }
    return ret;
  }

  // Templates for min/max expression matching
  private static Expression minmaxTemplates(MinMaxExpression e)
  {
    Set unique_children = new TreeSet(); // TreeSet uses compareTo method.

    unique_children.addAll(e.getChildren());

    MinMaxExpression ret = new MinMaxExpression(e.isMin());

    TreeMap literal_map = new TreeMap();

    // Separate literals and symbolic expressions.
    for ( Object child : unique_children )
    {
      if ( child instanceof IntegerLiteral )
        literal_map.put(new Long(child.toString()), child);
      else if ( child instanceof InfExpression )
      {
        //long inf = ((InfExpression)child).isMinus()?
        long inf = (((InfExpression)child).sign() < 0)?
          Long.MIN_VALUE:
          Long.MAX_VALUE;
        literal_map.put(new Long(inf), child);
      }
      else
        ret.add((Expression)child);
    }

    // Take only one literal child.
    if ( literal_map.size() > 0 )
    {
      if ( ret.isMin() )
        ret.add((Expression)literal_map.get(literal_map.firstKey()));
      else
        ret.add((Expression)literal_map.get(literal_map.lastKey()));
    }

    // Get out of min/max if it has only one child.
    if ( ret.getChildren().size() == 1 )
      return (Expression)ret.getChildren().get(0);

    // Match min(max(a,b),a) or max(min(a,b),a)
    if ( ret.getChildren().size() == 2 )
    {
      Expression e1 = (Expression)ret.getChildren().get(0);
      Expression e2 = (Expression)ret.getChildren().get(1);

      if ( e1 instanceof MinMaxExpression &&
        e1.getChildren().size() == 2 &&
        ((MinMaxExpression)e1).isMin() != ret.isMin() &&
        (e2.equals(e1.getChildren().get(0)) ||
        e2.equals(e1.getChildren().get(1))) )
        return e2;

      else if ( e2 instanceof MinMaxExpression &&
        e2.getChildren().size() == 2 &&
        ((MinMaxExpression)e2).isMin() != ret.isMin() &&
        ( e1.equals(e2.getChildren().get(0)) ||
        e1.equals(e2.getChildren().get(1)) ) )
        return e1;
    }

    // Pull out common min/max
    MinMaxExpression temp = new MinMaxExpression(ret.isMin());
    for ( Object child : ret.getChildren() )
    {
      if ( child instanceof MinMaxExpression )
      {
        MinMaxExpression mchild = (MinMaxExpression)child;
        if ( mchild.isMin() != ret.isMin() )
          return ret;
          //return new RangeExpression(
          //  new InfExpression(true), new InfExpression(false));

        for ( Object subchild : mchild.getChildren() )
          temp.add((Expression)subchild);
      }
      else
        temp.add((Expression)child);
    }
    return temp;
    //return ret;
  }

  // Template-based simplification in Expression object.
  private static Expression simpTemplates(Expression e)
  {
    Expression ret = (new NormalExpression(e)).getExpression();

    //System.out.println("simpTemplates "+ret);

    if ( ret instanceof BinaryExpression )
      ret = binaryTemplates((BinaryExpression)ret);

    else if ( ret instanceof MinMaxExpression )
      ret = minmaxTemplates((MinMaxExpression)ret);

    return ret;
  }

  // Binary shift simplification
  private static Expression simplifyShift(Expression e)
  {
    Expression ret = (Expression)e.clone();

    FlatIterator iter = new FlatIterator(ret);

    while ( iter.hasNext() )
    {
      Expression child = (Expression)iter.next();
      child.swapWith(simplifyShift(child));
    }

    if ( ret instanceof BinaryExpression )
    {
      BinaryExpression be = (BinaryExpression)e;
      BinaryOperator bop = be.getOperator();
      Expression lhs = be.getLHS(), rhs = be.getRHS();
      if ( lhs instanceof IntegerLiteral && rhs instanceof IntegerLiteral )
      {
        long lhs_value = ((IntegerLiteral)lhs).getValue();
        long rhs_value = ((IntegerLiteral)rhs).getValue();
        if ( bop == BinaryOperator.SHIFT_LEFT )
          ret = new IntegerLiteral(lhs_value << rhs_value);
        else if ( bop == BinaryOperator.SHIFT_RIGHT )
          ret = new IntegerLiteral(lhs_value >> rhs_value);
      }
    }
    return ret;
  }


  // Top-level simplify() 
  private NormalExpression simplify(String opt)
  {
    //System.out.println("IN simplify(opt): "+this);
    if ( op.equals(OP.UNIT) )
      return this;

    NormalExpression ret = new NormalExpression(this.op);

    for ( int i=0; i<list.size(); ++i )
      ret.addMerge(get(i).simplify(opt));
    // Synchronize before further simplification
    ret.sort();

    switch ( op )
    {
      case ADD:
        ret = ret.fold(opt);
        if ( opt.equals("FACT") )
          ret = ret.fact(opt);
        break;
      case MUL:
        ret = ret.dist(opt);
        ret = ret.fold(opt);
        break;
      case DIV: case MOD:
        ret = ret.div(opt);
        ret = ret.fold(opt);
        break;
      case NEG: case EQ: case NE: case GE: case GT: case LE: case LT: case AND:
      case OR:
        ret = ret.logic(opt);
        break;
      default:
    }
    return ret;
  }

  // Simplify Logical Expressions
  private NormalExpression logic(String opt)
  {
    if ( op.equals(OP.UNIT) || !(opt.equals("SIMP")||opt.equals("LOGIC")) )
      return this;

    NormalExpression ret = this;

    switch ( op )
    {
      case NEG:
        ret = get(0).negate();
        break;
      case EQ: case NE: case GE: case GT: case LE: case LT:
        ret = NormalRelation();
        ret = ret.simplifyRelation();
        break;
      case AND: case OR:
        ret = foldLogic();
        break;
      default:
    }
    return ret;
  }

  // Factorize expressions
  private NormalExpression fact(String opt)
  {
    if ( !op.equals(OP.ADD) || !(opt.equals("SIMP")||opt.equals("FACT")) )
      return this;

    NormalExpression gcd = null;

    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      gcd = (gcd==null)? curr: getGCD(gcd,curr).get(0);
      if ( gcd.compareTo(One) == 0 )
        return this;
    }

    NormalExpression rhs = new NormalExpression(OP.ADD);
    for ( Object o: list )
      rhs.add(getGCD((NormalExpression)o, gcd).get(1));

    // Just for keeping the normalized form
    rhs.sort();
    NormalExpression ret = new NormalExpression(OP.MUL);
    if ( gcd.op == OP.MUL )
    {
      ret.list.addAll(gcd.list);
      ret.order += gcd.order;
    }
    else
      ret.add(gcd);
    ret.add(rhs);
    ret.sort();
    return ret;
  }

  // Distribute expressions
  private NormalExpression dist(String opt)
  {
    if ( !op.equals(OP.MUL) || !(opt.equals("SIMP")||opt.equals("DIST")) )
      return this;

    NormalExpression ret = null;
    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      ret = (ret==null)? curr: dist(ret,curr);
    }
    return ret;
  }

  // Distribute expressions, a*(b+c) --> a*b+a*c
  private static NormalExpression dist
  (NormalExpression ne1, NormalExpression ne2)
  {
    // Normalize the forms
    NormalExpression e1 = join(getInteger(0), OP.ADD, ne1);
    NormalExpression e2 = join(getInteger(0), OP.ADD, ne2);

    NormalExpression ret = new NormalExpression(OP.ADD);

    for ( Object o1: e1.list )
      for ( Object o2: e2.list )
        ret.add(multiply((NormalExpression)o1,(NormalExpression)o2));

    return ret.fold("FOLD");
  }

  // Divide expressions
  private NormalExpression div(String opt)
  {
    if ( !op.equals(OP.DIV) && !op.equals(OP.MOD) ||
      !(opt.equals("SIMP")||opt.equals("DIV")) )
      return this;

    if ( op.equals(OP.MOD) )
      return mod();

    NormalExpression dividend = get(0).fact(opt);
    NormalExpression divisor = get(1).fact(opt);

    // Leave division-by-zero as it is
    if ( divisor.compareTo(Zero) == 0 )
      return this;
    // Simplify division-by-one
    if ( divisor.compareTo(One) == 0 )
      return get(0);

    NormalExpression temp = getGCD(dividend, divisor);
    NormalExpression ret = new NormalExpression(OP.DIV);
    ret.add(temp.get(1));
    ret.add(temp.get(2));

    // Now come literal simplifications
    Expression e1=ret.get(0).getExpression(), e2=ret.get(1).getExpression();
    Integer
      num1 = (e1 instanceof IntegerLiteral)? new Integer(e1.toString()): null,
      num2 = (e2 instanceof IntegerLiteral)? new Integer(e2.toString()): null;

    // Leave it with that if no change is required
    if ( num1 == null && num2 == null || num2 != null && num2.intValue() == 0 )
      return ret;
    // Numerator is int 0
    if ( num1 != null && num1.intValue() == 0 )
      return getInteger(0);
    // Integer division
    if ( num1 != null && num2 != null )
      return getInteger(num1.intValue()/num2.intValue());

    return ret;
  }

  // Fold constants and other terms
  private NormalExpression fold(String opt)
  {
    if ( !opt.equals("SIMP") && !opt.equals("FOLD") )
      return this;
    if ( op.equals(OP.ADD) )
      return foldADD();
    else if ( op.equals(OP.MUL) )
      return foldMUL();
    else
      return this;
  }

  private NormalExpression foldADD()
  {
    // Fold symbolic parts
    sort();
    NormalExpression ret = new NormalExpression(OP.ADD);
    NormalExpression prev = null, sum = null;

    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.order == 0 )
      {
        ret.add(curr);
        continue;
      }
      if ( prev == null )
      {
        sum = curr.getCoef();
        prev = curr;
      }
      else if ( prev.getVariable().compareTo(curr.getVariable()) != 0 )
      {
        ret.add(multiply(sum, prev.getVariable()));
        sum = curr.getCoef();
        prev = curr;
      }
      else
        sum = add(sum, curr.getCoef());

      if ( o == list.get(list.size()-1) )
        ret.add(multiply(sum, curr.getVariable()));
    }

    // Fold numeric parts
    prev = null;
    List temp = ret.list;
    ret = new NormalExpression(OP.ADD);
    for ( Object o: temp )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.isLiteral() )
        prev = (prev==null)? curr: add(prev, curr);
      else
        ret.add(curr);
    }
    // No numeric terms
    if ( prev == null )
      return (ret.list.size()==1)? ret.get(0): ret;
    // No symbolic terms
    if ( ret.list.size() == 0 )
      return prev;
    // Addition by zero
    if ( prev.toDouble() == 0 )
      return (ret.list.size()==1)? ret.get(0): ret;
    // Both kinds of terms
    ret.list.add(0, prev);
    return ret;
  }

  private NormalExpression foldMUL()
  {
    NormalExpression ret = new NormalExpression(OP.MUL), prev = null;
    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.isLiteral() )
        prev = (prev==null)? curr: multiply(prev, curr);
      else
        ret.add(curr);
    }
    // No numeric terms
    if ( prev == null )
      return (ret.list.size()==1)? ret.get(0): ret;
    // No symbolic terms or multiplication by zero
    if ( ret.list.size() == 0 || prev.toDouble() == 0 )
      return prev;
    // Multiplication by one
    if ( prev.toDouble() == 1 )
      return (ret.list.size()==1)? ret.get(0): ret;
    // Both kinds of terms
    ret.list.add(0, prev);
    return ret;
  }

  // Converts integer division to mod operations as much as possible
  private NormalExpression simplifyIntDiv()
  {
    if ( op.equals(OP.UNIT) )
      return this;

    for ( int i=0; i<list.size(); ++i )
      set(i, get(i).simplifyIntDiv());

    if ( !op.equals(OP.MUL) || type != INTEGER_ARITHMETIC )
      return this;

    NormalExpression factor = new NormalExpression(OP.MUL);
    NormalExpression ret = new NormalExpression(OP.MUL);
    List dividers = new ArrayList();

    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( curr.op.equals(OP.DIV) )
        dividers.add(curr);
      else
        factor.add(curr);
    }
    for ( Object o: dividers )
    {
      NormalExpression curr = (NormalExpression)o;
      NormalExpression gcd = getGCD(factor, curr.get(1));
      if ( gcd.get(0).compareTo(One) == 0 )
        continue;
      // Add remaining factors
      ret.addMerge(gcd.get(1));
      // Add converted expressions
      ret.add(add(curr.get(0), multiply(getInteger(-1),
        join(curr.get(0), OP.MOD, curr.get(1)))));
    }

    if ( ret.list.size() == 0 )
      return this;
    return ret.fold("FOLD");
  }

  // Simplifies MOD (%) operations
  private NormalExpression mod()
  {
    // Factorize for further simplification
    NormalExpression ne1 = get(0).fact("FACT"), ne2 = get(1).fact("FACT");
    NormalExpression ret = new NormalExpression(OP.MOD);
    ret.add(ne1); ret.add(ne2);

    Expression e1 = ne1.getExpression(), e2 = ne2.getExpression();
    Integer
      num1 = (e1 instanceof IntegerLiteral)? new Integer(e1.toString()): null,
      num2 = (e2 instanceof IntegerLiteral)? new Integer(e2.toString()): null;

    // Float numbers
    if ( e1 instanceof FloatLiteral || e2 instanceof FloatLiteral )
      return ret;

    // 0 divisor
    if ( num2 != null && num2 == 0 )
      return ret;

    // Compute % directly
    if ( num1 != null && num2 != null )
      return getInteger(num1%num2);
    
    NormalExpression gcd = getGCD(ne1, ne2);

    // No common divisor except for 1
    if ( gcd.get(0).compareTo(One) == 0 )
      return ret;
    // Divided evenly
    if ( gcd.get(2).compareTo(One) == 0 )
      return getInteger(0);

    // (a*x)%(b*x) = (a%b)*x
    NormalExpression mod = new NormalExpression(OP.MOD);
    mod.add(gcd.get(1)); mod.add(gcd.get(2));
    return multiply(gcd.get(0),mod).simplify("SIMP");
  }

  // Simplifies relations
  private NormalExpression simplifyRelation()
  {
    if ( !op.isCompare() )
      return this;

    if ( !get(0).isLiteral() || !get(1).isLiteral() )
      return this;

    int rel = get(0).toDouble().compareTo(get(1).toDouble());
    boolean result;
    switch ( rel )
    {
      case -1:
        result = ( op.equals(OP.LT) || op.equals(OP.LE) || op.equals(OP.NE) );
        break;
      case 0:
        result = ( op.equals(OP.LE) || op.equals(OP.EQ) || op.equals(OP.GE) );
        break;
      case 1:
      default:
        result = ( op.equals(OP.GT) || op.equals(OP.GE) || op.equals(OP.NE) );
    }

    if ( result )
      return getInteger(1);
    else
      return getInteger(0);
  }

  private NormalExpression negate()
  {
    NormalExpression ret = new NormalExpression();
    switch ( op )
    {
      case NEG:
        ret = get(0);
        break;
      case AND: case OR:
        ret.op = op.getNegation();
        for ( Object o: list )
          ret.add(((NormalExpression)o).negate());
        break;
      case EQ: case NE: case LT: case LE: case GT: case GE:
        ret = new NormalExpression(op.getNegation());
        ret.add(get(0)); ret.add(get(1));
        break;
      default:
        if ( isLiteral() )
        {
          ret = ( toDouble()==0 )? getInteger(1): getInteger(0);
          break;
        }
        ret.add(this);
        ret.op = OP.NEG;
    }
    return ret;
  }

  private NormalExpression foldLogic()
  {
    if ( !op.equals(OP.OR) && !op.equals(OP.AND) )
      return this;

    NormalExpression ret = new NormalExpression(op), prev = null;
    for ( Object o: list )
    {
      NormalExpression curr = (NormalExpression)o;
      if ( prev == null )
      {
        prev = curr;
        continue;
      }
      if ( op.equals(OP.OR) &&
        (prev.compareTo(One) == 0 || curr.compareTo(One) == 0) )
        return getInteger(1);
      if ( op.equals(OP.AND) &&
        (prev.compareTo(Zero) == 0 || curr.compareTo(Zero) ==0) )
        return getInteger(0);

      NormalExpression folded = foldLogic(prev,op,curr);
      if ( folded.op.equals(op) )
      {
        ret.add(prev);
        prev = curr;
      }
      else
        prev = folded;

      if ( o == list.get(list.size()-1) )
        ret.add(prev);
    }

    if ( ret.list.size() == 1 )
      return ret.get(0);
    else
      return ret;
  }

  // Minimizes AND operation if more hints are available. It works by
  // normalizing two relation expressions to a form which contains only zero
  // on the RHS. OR operation is also minimized by applying De Morgan's law.
  // 
  // a,b: expressions
  // (a op1 0) && (b op2 0) is rewritten as follows
  //
  // if a < b:
  //op1\op2  <    <=    ==    !=    >    >=
  //   <    b<0  b<=0  b==0   -     -    -
  //   <=   b<0  b<=0  b==0   -     -    -
  //   ==    F    F     F    a==0  a==0 a==0
  //   !=   b<0  b<=0  b==0   -     -    -
  //   >     F    F     F    a>0   a>0  a>0
  //   >=    F    F     F    a>=0  a>=0 a>=0
  //
  //
  // if a == b:
  //op1\op2  <    <=    ==    !=    >    >=
  //   <    a<0  a<0    F    a<0    F    F
  //   <=   a<0  a<=0  a==0  a<0    F   a==0
  //   ==    F   a==0  a==0   F     F   a==0
  //   !=   a<0  a<0    F    a!=0  a>0  a>0
  //   >     F    F     F    a>0   a>0  a>0
  //   >=    F   a==0  a==0  a>0   a>0  a>=0
  private static NormalExpression foldLogic
  (NormalExpression ne1, OP op, NormalExpression ne2)
  {
    NormalExpression e1 = ne1, e2 = ne2;
    if ( op.equals(OP.OR) )
    {
      e1 = ne1.negate();
      e2 = ne2.negate();
    }
    else if ( !op.equals(OP.AND) )
      return null;

    // Normalize TRUE literals
    if ( e1.isLiteral() && e1.toDouble() != 0 )
      e1 = getInteger(1);
    if ( e2.isLiteral() && e2.toDouble() != 0 )
      e2 = getInteger(1);

    NormalExpression ret = new NormalExpression(OP.AND);
    ret.add(e1); ret.add(e2);

    // Skip logical constant "TRUE"
    if ( e1.compareTo(One) == 0 )
      ret = e2;

    // Match e && e
    else if ( e2.compareTo(One) == 0 || e1.compareTo(e2) == 0 )
      ret = e1;

    // Match !e && e
    else if ( e1.negate().compareTo(e2) == 0 )
      ret = getInteger(0);

    if ( !e1.op.isCompare() || !e2.op.isCompare() )
      return ( op.equals(OP.OR) )? ret.negate(): ret;

    e1 = e1.NormalRelation();
    e2 = e2.NormalRelation();

    // Test if the two relation is reducible 
    NormalExpression diff = add(e1.get(0), e2.get(0));
    if ( diff.isLiteral() )
    {
      e2.set(0, multiply(getInteger(-1), e2.get(0)));
      e2.op = e2.op.getExchange();
    }
    diff = subtract(e1.get(0), e2.get(0));
    if ( !diff.isLiteral() )
      return ( op.equals(OP.OR) )? ret.negate(): ret;

    Double delta = diff.toDouble();
    
    NormalExpression a = (delta <= 0)? e1: e2;
    NormalExpression b = (delta > 0)? e1: e2;

    if ( delta == 0 )
    {
      switch ( a.op )
      {
        case LT:
          if ( b.op.equals(OP.EQ) || b.op.equals(OP.GT) || b.op.equals(OP.GE) )
            ret = getInteger(0);
          else
            ret = a;
          break;
        case LE:
          if ( b.op.equals(OP.LT) || b.op.equals(OP.NE) )
          {
            ret = a;
            ret.op = OP.LT;
          }
          else if ( b.op.equals(OP.LE) )
            ret = a;
          else if ( b.op.equals(OP.EQ) || b.op.equals(OP.GE) )
          {
            ret = a;
            ret.op = OP.EQ;
          }
          else
            ret = getInteger(0);
          break;
        case EQ:
          if ( b.op.equals(OP.LT) || b.op.equals(OP.NE) || b.op.equals(OP.GT) )
            ret = getInteger(0);
          else
            ret = a;
          break;
        case NE:
          if ( b.op.equals(OP.LT) || b.op.equals(OP.LE) )
          {
            ret = a;
            ret.op = OP.LT;
          }
          else if ( b.op.equals(OP.EQ) )
            ret = getInteger(0);
          else if ( b.op.equals(OP.NE) )
            ret = a;
          else
          {
            ret = a;
            ret.op = OP.GT;
          }
          break;
        case GT:
          if ( b.op.equals(OP.LT) || b.op.equals(OP.LE) || b.op.equals(OP.EQ) )
            ret = getInteger(0);
          else
            ret = a;
          break;
        case GE:
        default:
          if ( b.op.equals(OP.LT) )
            ret = getInteger(0);
          else if ( b.op.equals(OP.LE) || b.op.equals(OP.EQ) )
          {
            ret = a;
            ret.op = OP.EQ;
          }
          else if ( b.op.equals(OP.NE) || b.op.equals(OP.GT) )
          {
            ret = a;
            ret.op = OP.GT;
          }
          else
            ret = a;
      }
    }

    else
    {
      if ( a.op.equals(OP.EQ) || a.op.equals(OP.GT) || a.op.equals(OP.GE) )
      {
        if ( b.op.equals(OP.NE) || b.op.equals(OP.GT) || b.op.equals(OP.GE) )
          ret = a;
        else
          ret = getInteger(0);
      }
      else
      {
        if ( b.op.equals(OP.NE) || b.op.equals(OP.GT) || b.op.equals(OP.GE) )
          ;
        else
          ret = b;
      }
    }
    return ( op.equals(OP.OR) )? ret.negate(): ret;
  }

  private enum OP
  {
    UNIT, NEG, ADD, MUL, AND, OR, EQ, NE, GE, GT, LE, LT, DIV, MOD;
    boolean isCommAssoc()
      { return ( compareTo(ADD)>=0 && compareTo(OR)<=0 ); }
    boolean isCompare()
      { return ( compareTo(EQ)>=0 && compareTo(LT)<=0 ); }
    boolean isLogical()
      { return ( compareTo(AND)>=0 && compareTo(LT)<=0 ); }
    OP getNegation()
    {
      switch ( this )
      {
        case EQ: return NE; case NE: return EQ; case GE: return LT;
        case GT: return LE; case LE: return GT; case LT: return GE;
        case AND: return OR; case OR: return AND;
        default: return this;
      }
    }
    OP getExchange()
    {
      switch ( this )
      {
        case GE: return LE; case GT: return LT; case LE: return GE;
        case LT: return GT; default: return this;
      }
    }
    Printable getOP()
    {
      switch ( this )
      {
        case NEG: return UnaryOperator.LOGICAL_NEGATION;
        case ADD: return BinaryOperator.ADD;
        case MUL: return BinaryOperator.MULTIPLY;
        case AND: return BinaryOperator.LOGICAL_AND;
        case OR:  return BinaryOperator.LOGICAL_OR;
        case EQ:  return BinaryOperator.COMPARE_EQ;
        case NE:  return BinaryOperator.COMPARE_NE;
        case GE:  return BinaryOperator.COMPARE_GE;
        case GT:  return BinaryOperator.COMPARE_GT;
        case LE:  return BinaryOperator.COMPARE_LE;
        case LT:  return BinaryOperator.COMPARE_LT;
        case DIV: return BinaryOperator.DIVIDE;
        case MOD: return BinaryOperator.MODULUS;
        default:  return null;
      }
    }
    static OP setOP(Printable op)
    {
      for ( OP ops: OP.values() )
        if ( ops.getOP() == op )
          return ops;
      return UNIT;
    }
  }

	// Test routines.
	/*
	public static void main(String[] args)
	{
		Program program = cetus.exec.Driver.getProgram(args);

    DepthFirstIterator iter = new DepthFirstIterator(program);

		while ( iter.hasNext() )
		{
			Object o = iter.next();

			if ( o instanceof ExpressionStatement )
			{
	      Expression e = ((ExpressionStatement)o).getExpression();
				System.out.print(e.toString()+":\n");

      	//e.print(System.out);
      	//System.out.println(":");

      	String options[] = {"FOLD","DIST","DIV","FACT","LOGIC","SIMP"};
      	for ( String opt: options )
        	System.out.printf("%5s -> %s\n", opt, simplify(e,opt).toString());

      	System.out.println("");
			}
		}
	}
	*/

}
