package cetus.hir;

import java.util.*;

/**
 * Iterates over Traversable objects in depth-first order.
 */
public class DepthFirstIterator extends IRIterator
{
  private Vector<Traversable> stack;
  private HashSet<Class> prune_set;

  /**
   * Creates a new iterator.
   *
   * @param init The first object to visit.
   */
  public DepthFirstIterator(Traversable init)
  {
    super(init);
    stack = new Vector<Traversable>();
    stack.add(init);
    prune_set = new HashSet<Class>();
  }

  public boolean hasNext()
  {
    return !stack.isEmpty();
  }

  public Object next()
  {
    Traversable t = null;

    try {
      t = stack.remove(0);
    } catch (EmptyStackException e) {
      throw new NoSuchElementException();
    }

/*
    if (t.getChildren() != null
        && !containsCompatibleClass(prune_set, t.getClass()))
*/
		if ( !containsCompatibleClass(prune_set, t.getClass()) &&
			t.getChildren() != null )
    {
      int i = 0;
			/*
      Iterator iter = t.getChildren().iterator();
      while (iter.hasNext())
      {
        Object o = iter.next();
        if (o != null)
          stack.add(i++, o);
      }
			*/
			for(Traversable o : t.getChildren())
        if (o != null)
          stack.add(i++, o);
				
    }

    return t;
  }

  public void pruneOn(Class c)
  {
    prune_set.add(c);
  }

	/**
		* Returns a linked list of objects of Class c in the IR
		*/
	public ArrayList getList(Class c)
	{
		ArrayList list = new ArrayList();

		while (hasNext())
		{
			Object obj = next();
			if (c.isInstance(obj))
			{
				list.add(obj);
			}
		}
		return list;
	}

	/**
		* Returns a set of objects of Class c in the IR
		*/
	public Set getSet(Class c)
	{
		HashSet set = new HashSet();

		System.out.println("getSet strt");
		while (hasNext())
		{
			Object obj = next();

			System.out.println(obj.getClass().toString() + ": " + obj.toString());

      if (c.isInstance(obj))
			{
				set.add(obj);
			}
		}
		return set;
	}
 
  public void reset()
  {
    stack.clear();
    stack.add(root);
  }
}
