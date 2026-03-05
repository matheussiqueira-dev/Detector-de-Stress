import type { ButtonHTMLAttributes, ReactNode } from "react";

import styles from "./TronButton.module.css";

type TronButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
};

export function TronButton({ children, className, type = "button", ...props }: TronButtonProps) {
  return (
    <button className={[styles.button, className].filter(Boolean).join(" ")} type={type} {...props}>
      {children}
    </button>
  );
}
