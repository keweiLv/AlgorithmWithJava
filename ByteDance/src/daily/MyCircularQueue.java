package daily;

/**
 * @author Kezi
 * @date 2022年08月02日 22:41
 * 循环队列
 */
public class MyCircularQueue {

	int k, he, ta;
	int[] nums;

	public MyCircularQueue(int k) {
		this.k = k;
		nums = new int[this.k];
	}

	public boolean enQueue(int value) {
		if (isFull()) {
			return false;
		}
		nums[ta % k] = value;
		return ++ta >= 0;
	}

	public boolean deQueue() {
		if (isEmpty()) {
			return false;
		}
		return ++he >= 0;
	}

	public int Front() {
		return isEmpty() ? -1 : nums[he % k];
	}

	public int Rear() {
		return isEmpty() ? -1 : nums[(ta - 1) % k];
	}

	public boolean isEmpty() {
		return he == ta;
	}

	public boolean isFull() {
		return ta - he == k;
	}
}
