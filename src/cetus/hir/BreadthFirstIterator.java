package cetus.hir;

import java.util.*;

/**
 * Iterates over Traversable objects in breadth-first order.
 */
public class BreadthFirstIterator extends IRIterator
{
  private Vector<Traversable> queue;
  private HashSet<Class> prune_set;

  /**
   * Creates a new iterator.
   *
   * @param init The first object to visit.
   */
  public BreadthFirstIterator(Traversable init)
  {
    super(init);
    queue = new Vector<Traversable>();
    queue.add(init);
    prune_set = new HashSet<Class>();
  }

  public boolean hasNext()
  {
    return !queue.isEmpty();
  }

  public Object next()
  {
    Traversable t = null;

    try {
      t = queue.remove(0);
    } catch (ArrayIndexOutOfBoundsException e) {
      throw new NoSuchElementException();
    }

    if (t.getChildren() != null
        && !containsCompatibleClass(prune_set, t.getClass()))
    {
		/*
      Iterator iter = t.getChildren().iterator();
      while (iter.hasNext())
      {
        Object o = iter.next();
        if (o != null)
          queue.add(o);
      }
		*/
			for(Traversable o : t.getChildren())
				if(o != null)
					queue.add(o);
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
    queue.clear();
    queue.add(root);
  }
}
