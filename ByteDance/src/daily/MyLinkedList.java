package daily;

/**
 * @author Kezi
 * @date 2022年09月23日 23:05
 * @deprecated 设计链表
 */
public class MyLinkedList {
	class Node {
		Node pre, next;
		int val;

		Node(int _val) {
			val = _val;
		}
	}

	Node he = new Node(-1), ta = new Node(-1);
	int size = 0;

	public MyLinkedList() {
		he.next = ta;
		ta.pre = he;
	}

	public int get(int index) {
		Node node = getNode(index);
		return node == null ? -1 : node.val;
	}

	private Node getNode(int index) {
		boolean isLeft = index < size / 2;
		if (!isLeft){
			index = size - index -1;
		}
		for (Node cur = isLeft ? he.next : ta.pre; cur != ta && cur != he; cur = isLeft ? cur.next : cur.pre) {
		int tmp = cur.val;
			if (index-- == 0){
				return cur;
			}
		}
		return null;
	}

	public void addAtHead(int val) {
		Node node = new Node(val);
		node.next = he.next;
		node.pre = he;
		he.next.pre = node;
		he.next = node;
		size++;
	}

	public void addAtTail(int val) {
		Node node = new Node(val);
		node.pre = ta.pre;
		node.next = ta;
		ta.pre.next = node;
		ta.pre = node;
		size++;
	}

	public void addAtIndex(int index, int val) {
		if (index > size){
			return;
		}
		if (index <= 0){
			addAtHead(val);
		}else if (index == size){
			addAtTail(val);
		}else {
			Node node = new Node(val),cur = getNode(index);
			node.next = cur;
			node.pre= cur.pre;
			cur.pre.next = node;
			cur.pre = node;
			size++;
		}
	}

	public void deleteAtIndex(int index) {
		Node cur = getNode(index);
		if (cur == null){
			return;
		}
		cur.next.pre = cur.pre;
		cur.pre.next = cur.next;
		size--;
	}

}
