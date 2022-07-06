package daily;

import java.util.TreeSet;

/**
 * @author Kezi
 * @date 2022年07月05日 22:35
 * 我的日程安排表Ⅰ
 */
public class MyCalendar {

	TreeSet<int[]> booked;

	public MyCalendar() {
		booked = new TreeSet<int[]>((a, b) -> a[0] - b[0]);
	}

	public boolean book(int start, int end) {
		if (booked.isEmpty()) {
			booked.add(new int[]{start, end});
			return true;
		}
		int[] tmp = {end, 0};
		int[] arr = booked.ceiling(tmp);
		int[] prev = arr == null ? booked.last() : booked.lower(arr);
		if (arr == booked.first() || booked.lower(tmp)[1] <= start) {
			booked.add(new int[]{start, end});
			return true;
		}
		return false;
	}
}
