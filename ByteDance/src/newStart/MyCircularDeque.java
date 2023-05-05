package newStart;

/**
 * 设计循环双端队列
 */
public class MyCircularDeque {

    int[] nums;
    int he, ta, cnt, k;

    public MyCircularDeque(int _k) {
        k = _k;
        nums = new int[k];
    }

    public boolean insertFront(int value) {
        if (isFull()) {
            return false;
        }
        he = (he + k - 1) % k;
        nums[he] = value;
        cnt++;
        return true;
    }

    public boolean insertLast(int value) {
        if (isFull()) {
            return false;
        }
        nums[ta++] = value;
        cnt++;
        ta %= k;
        return true;
    }

    public boolean deleteFront() {
        if (isEmpty()) {
            return false;
        }
        he = (he + 1) % k;
        cnt--;
        return true;
    }

    public boolean deleteLast() {
        if (isEmpty()) {
            return false;
        }
        ta = (ta + k - 1) % k;
        cnt--;
        return true;
    }

    public int getFront() {
        return isEmpty() ? -1 : nums[he];
    }

    public int getRear() {
        return isEmpty() ? -1 : nums[(ta + k - 1) % k];
    }

    public boolean isEmpty() {
        return cnt == 0;
    }

    public boolean isFull() {
        return cnt == k;
    }
}
