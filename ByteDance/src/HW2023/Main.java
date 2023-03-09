package HW2023;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Scanner;

/**
 * @author Kezi
 * @date 2023年03月09日 23:47
 */
public class Main {
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		String key = in.nextLine();
		String str = in.nextLine();
		LinkedHashSet<Character> set = new LinkedHashSet<>();
		for (char c : key.toCharArray()){
			set.add(c);
		}
		int k = 0;
		while (set.size() < 26){
			set.add((char) ('a' + k));
			k++;
		}
		List<Character> list = new ArrayList<>(set);
		StringBuilder sb = new StringBuilder();
		for (int i = 0;i<str.length();i++){
			sb.append(list.get(str.charAt(i) - 'a'));
		}
		System.out.println(sb.toString());
	}
}
