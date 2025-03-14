// Type declarations for microbit-web-bluetooth
declare module 'microbit-web-bluetooth' {
  export class MicrobitBluetooth {
    connectAsync(): Promise<any>;
    disconnect(): void;
    onDisconnected(callback: () => void): void;
    getServices(): {
      ledService: {
        writeMatrixState(matrix: boolean[][]): Promise<void>;
        writeText(text: string): Promise<void>;
        setScrollingDelay(delay: number): Promise<void>;
      };
      buttonService: {
        getButtonAState(): Promise<number>;
        getButtonBState(): Promise<number>;
        onButtonAStateChanged(callback: (state: number) => void): void;
        onButtonBStateChanged(callback: (state: number) => void): void;
      };
      uartService: {
        sendString(text: string): Promise<void>;
        onReceiveString(callback: (text: string) => void): void;
      };
    };
  }
} 